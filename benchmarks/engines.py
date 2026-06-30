"""Engine adapters for the unified benchmark.

Each adapter wraps one transcription path (a starling megakernel pipeline, the
stock ``transformers`` reference, or the external CrispASR binary) behind one
uniform interface so :mod:`bench_all` can drive every model x engine x length x
batch cell of the grid with the same code.

The interface is intentionally tiny:

    name            -- display label (e.g. "stock transformers")
    supports_batch  -- True if ``transcribe`` honours B>1 in one fused call
    transcribe(audio, *, B) -> list[str]
                    -- transcribe ``audio`` (1-D float32 @16kHz) ``B`` times and
                       return ``B`` decoded strings. Batched engines do it in one
                       call; non-batched engines loop B times (the harness labels
                       that "Bx1 sequential").
    close()         -- free the model + empty the GPU cache

The adapters are LAZY: importing this module costs nothing; the heavy ``torch``
/ ``transformers`` imports happen inside :meth:`load`.

Adapter families live under :data:`ENGINE_REGISTRY`, keyed by
``"{family}-{model}"`` (e.g. ``"starling-granite"``, ``"stock-parakeet"``,
``"crispasr-granite"``). :func:`build_engines` resolves a list of
``--engines``/``--models`` filters into the concrete adapter objects.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


class SkipCell(Exception):
    """Raised when a (model, engine, length, batch) cell is infeasible.

    The harness catches it, records the cell as skipped with the reason, and
    continues the sweep -- so one infeasible cell (e.g. granite single-shot on
    audio longer than its static KV cache) never aborts the whole grid.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

# CrispASR install (absent -> crispasr-* engines are silently skipped).
ASR_BENCH = Path("/home/m0hawk/asr-bench")
CRISPASR_BIN = ASR_BENCH / "bin" / "crispasr-linux-x86_64-cuda13" / "crispasr"
CRISPASR_MODELS = ASR_BENCH / "models"
_CRISPASR_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": str(Path.home()),
    "LD_LIBRARY_PATH": (
        f"{ASR_BENCH}/libs/usr/lib/x86_64-linux-gnu/openblas-pthread"
        f":{ASR_BENCH}/bin/parakeet-v0.3.2-bin-linux-cuda-x64"
    ),
    "CRISPASR_N_GPU_LAYERS": "999",  # full GPU offload
}

# parakeet.cpp (mudler's C++/ggml parakeet-cli). The binary needs glibc 2.38
# which the host lacks, so it runs through the ld-linux shim wrapper. Absent
# install -> the parakeet_cpp engine is silently skipped.
PARAKEET_CPP_WRAP = ASR_BENCH / "parakeet-cli-wrap"
PARAKEET_CPP_MODEL = CRISPASR_MODELS / "tdt-0.6b-v3-f16.gguf"
_PARAKEET_CPP_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": str(Path.home()),
    "LD_LIBRARY_PATH": f"{ASR_BENCH}/bin/parakeet-v0.3.2-bin-linux-cuda-x64",
}


class Engine:
    """Base adapter. Subclasses override :meth:`load` / :meth:`_run_one`."""

    def __init__(self, name: str, model: str, *, supports_batch: bool = False) -> None:
        self.name = name          # display label
        self.model = model        # model slug (granite/parakeet/moss/qwen3)
        self.supports_batch = supports_batch
        self._loaded = False

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> None:
        """Load the model/processor (heavy). Idempotent."""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def close(self) -> None:
        """Release the model + drop the GPU cache. Safe to call repeatedly."""
        self._release()
        self._loaded = False
        torch.cuda.empty_cache()

    def _release(self) -> None:
        pass

    # -- inference ---------------------------------------------------------
    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, *, B: int = 1) -> list[str]:
        """Transcribe ``audio`` ``B`` times; return ``B`` decoded strings."""
        self.load()
        if self.supports_batch and B > 1:
            return self._run_batch([audio] * B)
        return [self._run_one(audio) for _ in range(B)]

    def _run_one(self, audio: np.ndarray) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _run_batch(self, audio_list: list[np.ndarray]) -> list[str]:
        # Default: loop (engines that truly batch override this).
        return [self._run_one(a) for a in audio_list]


# ====================================================================== #
# Granite-Speech-4.1-2b
# ====================================================================== #
class GraniteStarling(Engine):
    """starling fused megakernel pipeline (cudagraph encoder + K-step LLM).

    Uses the chunked ``transcribe_long`` path (the production path the README
    numbers use): it resets the static KV cache per chunk so peak VRAM is
    constant and any audio length is transcribable. Short audio = 1 chunk;
    longer audio = several chunks concatenated.
    """

    def __init__(self) -> None:
        super().__init__("starling", "granite", supports_batch=False)

    def _load(self) -> None:
        from starling.granite.pipeline import MegaPipeline

        self.pipe = MegaPipeline.from_pretrained()

    def _release(self) -> None:
        self.pipe = None

    def _run_one(self, audio: np.ndarray) -> str:
        from starling.granite.long_audio import transcribe_long

        wav = torch.from_numpy(audio).float().unsqueeze(0)
        res = transcribe_long(
            self.pipe, self.pipe.processor, wav, 16000,
            speculative=False,  # greedy, matching the non-spec "starling" column
        )
        return res.text


class GraniteStarlingSpec(Engine):
    """starling fused pipeline, **self-speculative** decode (granite only).

    The speculative companion to :class:`GraniteStarling`: drafts tokens from
    the encoder's CTC head and verifies them with the LLM in multi-token
    forwards. Appears as the ``starling (spec)`` column so the latency table
    carries the speculative-vs-greedy comparison the README describes (spec is
    slower on short audio, faster on long).
    """

    def __init__(self) -> None:
        super().__init__("starling (spec)", "granite", supports_batch=False)

    def _load(self) -> None:
        from starling.granite.pipeline import MegaPipeline

        self.pipe = MegaPipeline.from_pretrained()

    def _release(self) -> None:
        self.pipe = None

    def _run_one(self, audio: np.ndarray) -> str:
        from starling.granite.long_audio import transcribe_long

        wav = torch.from_numpy(audio).float().unsqueeze(0)
        res = transcribe_long(
            self.pipe, self.pipe.processor, wav, 16000, speculative=True,
        )
        return res.text


class GraniteStarlingBatched(Engine):
    """starling fused pipeline, **batched** LLM decode (B chunks in lock-step).

    The companion to :class:`GraniteStarling`: same encoder/projector per
    stream (byte-exact), but the K-step LLM decode is batched via
    :class:`~starling.granite.batched.BatchedPipeline`, turning the launch-bound
    batch=1 GEMVs into saturating B-wide GEMMs. Drives
    :func:`transcribe_long_batched`, so it handles arbitrary-length audio
    (chunks are grouped B-at-a-time; the last partial group is padded with
    copies of chunk 0). Appears as the ``starling (batched)`` engine label so
    the latency table shows a B>1 row for granite.
    """

    def __init__(self) -> None:
        super().__init__("starling (batched)", "granite", supports_batch=True)

    def _load(self) -> None:
        from starling.granite.batched import BatchedPipeline
        from starling.granite.loader import load_model_and_processor

        # BatchedPipeline needs the raw model + processor (not MegaPipeline).
        self.model, self.processor = load_model_and_processor(attn_impl="eager")
        self._max_B = 8  # rebuilt per batch in _run_batch via max_batch_size

    def _release(self) -> None:
        self.model = self.processor = None

    def _run_batch(self, audio_list: list[np.ndarray]) -> list[str]:
        from starling.granite.batched import BatchedPipeline
        from starling.granite.long_audio import transcribe_long_batched

        B = len(audio_list)
        # BatchedPipeline is statically sized to max_batch_size; rebuild per B
        # (capture is cheap relative to a timed run). Pad partial batches with
        # copies of stream 0 so B == max_batch_size exactly.
        pipe = BatchedPipeline(self.model, self.processor, max_batch_size=B)
        wav = torch.from_numpy(audio_list[0]).float().unsqueeze(0)
        # transcribe_long_batched decodes ONE clip; to time B independent clips
        # we tile the input B times into one B-chunk "long" audio. All B chunks
        # are identical (same transcript), so the per-stream RTFx is the batched
        # throughput RTFx -- matching how the README batched numbers are derived.
        tiled = wav.repeat(1, B) if B > 1 else wav
        res = transcribe_long_batched(pipe, self.processor, tiled, 16000)
        text = res.text.strip()
        del pipe
        torch.cuda.empty_cache()
        # transcribe_long_batched concatenates chunk texts with spaces; the B
        # identical chunks all produced `text` -> replicate it B times.
        return [text] * B


class GraniteStock(Engine):
    """Unmodified HuggingFace ``model.generate`` reference (chunked long path).

    Chunked like the starling path so it is comparable on every tier (single-shot
    stock on medium/long audio is both wrong -- RoPE positions -- and slow). To
    keep WER byte-comparable with starling we decode the GENERATED tokens only
    (slicing past the prompt), mirroring ``transcribe_long`` -- the repo's own
    ``transcribe_long_stock`` decodes prompt+generated, which would leak the
    task-prompt words into the transcript.
    """

    def __init__(self) -> None:
        super().__init__("stock transformers", "granite", supports_batch=False)

    def _load(self) -> None:
        from starling.granite.audio import build_inputs
        from starling.granite.loader import load_model_and_processor

        self._build_inputs = build_inputs
        self.model, self.processor = load_model_and_processor(attn_impl="eager")

    def _release(self) -> None:
        self.model = self.processor = None

    def _run_one(self, audio: np.ndarray) -> str:
        from starling.granite.long_audio import chunk_audio

        wav = torch.from_numpy(audio).float().unsqueeze(0)
        texts: list[str] = []
        for chunk_wav, _start, _end, _idx in chunk_audio(wav, 16000):
            inp = self._build_inputs(self.processor, chunk_wav)
            prompt_len = int(inp["input_ids"].shape[1])
            gen = self.model.generate(
                input_ids=inp["input_ids"],
                input_features=inp["input_features"].bfloat16(),
                attention_mask=inp["attention_mask"],
                input_features_mask=inp.get("input_features_mask"),
                max_new_tokens=200,
                do_sample=False,
                num_beams=1,
            )
            gen_new = gen[:, prompt_len:]
            texts.append(
                self.processor.tokenizer.batch_decode(
                    gen_new, skip_special_tokens=True
                )[0]
            )
        return " ".join(t.strip() for t in texts if t.strip())


# ====================================================================== #
# Parakeet-tdt-0.6b-v3
# ====================================================================== #
class ParakeetStarling(Engine):
    """starling GPU megakernel pipeline (GPU mel + graphed encoder + graphed TDT)."""

    def __init__(self) -> None:
        super().__init__("starling", "parakeet", supports_batch=True)

    def _load(self) -> None:
        from starling.parakeet.pipeline import MegaParakeetPipeline

        self.pipe = MegaParakeetPipeline()

    def _release(self) -> None:
        self.pipe = None

    def _run_one(self, audio: np.ndarray) -> str:
        return self.pipe.transcribe([audio])[0]

    def _run_batch(self, audio_list: list[np.ndarray]) -> list[str]:
        return self.pipe.transcribe(audio_list)


class ParakeetStock(Engine):
    """Stock ``AutoModelForTDT.generate`` reference (BaselineRunner)."""

    def __init__(self) -> None:
        super().__init__("stock transformers", "parakeet", supports_batch=False)

    def _load(self) -> None:
        from starling.granite.baseline import BaselineRunner

        self.runner = BaselineRunner()

    def _release(self) -> None:
        self.runner = None

    def _run_one(self, audio: np.ndarray) -> str:
        return self.runner.transcribe_batch([audio])[0]


# ====================================================================== #
# MOSS-Transcribe-preview-2b
# ====================================================================== #
class MossStarling(Engine):
    """starling fused pipeline (graphed encoder + K-step graphed LLM decode)."""

    def __init__(self) -> None:
        super().__init__("starling", "moss", supports_batch=False)

    def _load(self) -> None:
        from starling.moss.pipeline import MossMegaPipeline

        # max_cache_len=2048 (like benchmarks/moss/bench_pipeline.py) so the long
        # fixture (~470 generated tokens) fits the static KV cache.
        self.pipe = MossMegaPipeline.from_pretrained(max_cache_len=2048)

    def _release(self) -> None:
        self.pipe = None

    def _run_one(self, audio: np.ndarray) -> str:
        from starling.moss.loader import load_model_and_processor

        inp = self.pipe.processor(audio.astype("float32"))
        inp = {
            k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in inp.items()
        }
        text, _ = self.pipe.transcribe(
            inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
            inp["audio_input_mask"], max_new_tokens=400,
        )
        return text


class MossStock(Engine):
    """Stock eager greedy reference.

    MOSS's HF ``generate`` is broken on this transformers build's strict kwarg
    validation, so the byte-exact stock reference is the hand-written eager
    greedy loop in ``starling.moss.reference`` (it calls the identical model
    modules). See that module's docstring.
    """

    def __init__(self) -> None:
        super().__init__("stock transformers", "moss", supports_batch=False)

    def _load(self) -> None:
        from starling.moss.loader import load_model_and_processor
        from starling.moss.reference import (
            audio_features,
            build_inputs_embeds,
            greedy_generate,
        )

        self.model, self.processor = load_model_and_processor()
        self._audio_features = audio_features
        self._build_inputs_embeds = build_inputs_embeds
        self._greedy_generate = greedy_generate

    def _release(self) -> None:
        self.model = self.processor = None

    def _run_one(self, audio: np.ndarray) -> str:
        inp = self.processor(audio.astype("float32"))
        inp = {
            k: (v.cuda() if isinstance(v, torch.Tensor) else v)
            for k, v in inp.items()
        }
        feats = self._audio_features(
            self.model, inp["audio_data"], inp["audio_data_seqlens"]
        )
        emb = self._build_inputs_embeds(
            self.model, inp["input_ids"], feats, inp["audio_input_mask"]
        )
        ids = self._greedy_generate(self.model, emb, max_new_tokens=400, max_cache_len=2048)
        return self.processor.tokenizer.decode(ids[0], skip_special_tokens=True)


# ====================================================================== #
# Qwen3-ASR-1.7b  (auto-enabled once the branch is merged onto master)
# ====================================================================== #
class Qwen3Starling(Engine):
    """starling fused pipeline for Qwen3-ASR. Lives on the qwen3-asr branch."""

    def __init__(self) -> None:
        super().__init__("starling", "qwen3", supports_batch=False)

    def _load(self) -> None:
        from starling.qwen3.audio import build_inputs, load_wav  # noqa: F401
        from starling.qwen3.pipeline import MegaPipeline

        self._build_inputs = build_inputs
        self.pipe = MegaPipeline.from_pretrained()

    def _release(self) -> None:
        self.pipe = None

    def _run_one(self, audio: np.ndarray) -> str:
        import torchaudio

        wav = torch.from_numpy(audio).float().unsqueeze(0)
        inp = self._build_inputs(self.pipe.processor, wav, sr=16000)
        text, _ = self.pipe.transcribe(
            inp["input_features"], inp["input_ids"],
            inp.get("input_features_mask"), max_new_tokens=400,
        )
        return text


class Qwen3StarlingBatched(Engine):
    """starling fused pipeline for Qwen3-ASR, **batched** decode.

    Single-shot (no chunker) batched decode via
    :class:`~starling.qwen3.batched.BatchedPipeline`: each of the B clips is
    encoded byte-exactly (batch=1), then all B are decoded in one lock-step
    pass over a shared static KV cache. The pipeline is statically sized to
    ``max_batch_size == B``, so partial batches are padded with copies of
    stream 0. Bounded by the 4096-token cache, so batched cells only appear
    on tiers whose prompt+output fits the cache (short/medium).
    """

    def __init__(self) -> None:
        super().__init__("starling (batched)", "qwen3", supports_batch=True)

    def _load(self) -> None:
        from starling.qwen3.audio import build_inputs
        from starling.qwen3.loader import load_model_and_processor

        self._build_inputs = build_inputs
        self.model, self.processor = load_model_and_processor()
        self._pipe = None

    def _release(self) -> None:
        self.model = self.processor = self._pipe = None

    def _run_batch(self, audio_list: list[np.ndarray]) -> list[str]:
        from starling.qwen3.batched import BatchedPipeline

        B = len(audio_list)
        # Pipeline is statically sized to B; rebuild per batch size (capture is
        # cheap relative to a timed run). Pad short batches with stream-0 copies
        # so len == max_batch_size exactly (transcribe_batch raises otherwise).
        feats, ids, masks = [], [], []
        for a in audio_list:
            wav = torch.from_numpy(a).float().unsqueeze(0)
            inp = self._build_inputs(self.processor, wav, sr=16000)
            feats.append(inp["input_features"])
            ids.append(inp["input_ids"])
            masks.append(inp.get("input_features_mask"))
        pipe = BatchedPipeline(self.model, self.processor,
                               max_batch_size=B, max_cache_len=4096)
        texts = pipe.transcribe_batch(feats, ids, masks, max_new_tokens=400)
        del pipe
        torch.cuda.empty_cache()
        return texts


class Qwen3Stock(Engine):
    """Stock ``model.generate`` reference for Qwen3-ASR (branch qwen3-asr)."""

    def __init__(self) -> None:
        super().__init__("stock transformers", "qwen3", supports_batch=False)

    def _load(self) -> None:
        from starling.qwen3.audio import build_inputs
        from starling.qwen3.loader import load_model_and_processor

        self._build_inputs = build_inputs
        self.model, self.processor = load_model_and_processor(attn_impl="eager")

    def _release(self) -> None:
        self.model = self.processor = None

    def _run_one(self, audio: np.ndarray) -> str:
        wav = torch.from_numpy(audio).float().unsqueeze(0)
        inp = self._build_inputs(self.processor, wav, sr=16000)
        prompt_len = inp["input_ids"].shape[1]
        gen = self.model.generate(
            input_ids=inp["input_ids"],
            input_features=inp["input_features"],
            input_features_mask=inp.get("input_features_mask"),
            max_new_tokens=400,
            do_sample=False,
            num_beams=1,
        )
        gen_new = gen[:, prompt_len:]
        try:
            return self.processor.decode(gen_new, return_format="transcription_only")[0]
        except Exception:  # noqa: BLE001 -- older processor API
            return self.processor.batch_decode(gen_new, skip_special_tokens=True)[0]


# ====================================================================== #
# ARK-ASR-3B
# ====================================================================== #
class ArkStarling(Engine):
    """starling fused pipeline for ARK-ASR-3B (graphed Whisper+adapter encoder
    + K-step graphed Qwen2.5 decode). Byte-identical to eager."""

    def __init__(self) -> None:
        super().__init__("starling", "ark", supports_batch=False)

    def _load(self) -> None:
        from starling.ark.pipeline import MegaPipeline

        self.pipe = MegaPipeline.from_pretrained()

    def _release(self) -> None:
        self.pipe = None

    def _run_one(self, audio: np.ndarray) -> str:
        text, _ids = self.pipe.transcribe(audio, max_new_tokens=200)
        return text


class ArkStock(Engine):
    """Stock eager ``model.generate`` reference for ARK-ASR-3B.

    Drives the chat-template prompt the processor builds (audio block +
    instruction) through ``AutoModelForCausalLM.generate`` — the same path
    ``scripts/make_ark_golden.py`` captures the golden reference with.
    """

    def __init__(self) -> None:
        super().__init__("stock transformers", "ark", supports_batch=False)

    def _load(self) -> None:
        from starling.ark.config import DEFAULT_INSTRUCTION
        from starling.ark.loader import load_model_and_processor

        self._instr = DEFAULT_INSTRUCTION
        self.model, self.processor = load_model_and_processor(attn_impl="eager")

    def _release(self) -> None:
        self.model = self.processor = None

    def _run_one(self, audio: np.ndarray) -> str:
        wav = np.ascontiguousarray(audio, dtype=np.float32)
        conv = [{"role": "user", "content": [
            {"type": "audio", "array": wav},
            {"type": "text", "text": self._instr},
        ]}]
        data = self.processor.apply_chat_template(
            conv, audio_torch_dtype=torch.bfloat16, tokenize=True,
            return_tensors="pt", add_generation_prompt=True,
        )
        data = {k: v.to("cuda") for k, v in data.items()}
        prompt_len = data["input_ids"].shape[1]
        out = self.model.generate(**data, max_new_tokens=200, do_sample=False)
        gen = out[0][prompt_len:]
        return self.processor.tokenizer.decode(gen, skip_special_tokens=True)


# ====================================================================== #
# cohere-transcribe-03-2026  (seq2seq encoder-decoder)
# ====================================================================== #
class CohereStarling(Engine):
    """starling fused pipeline for cohere-transcribe (graphed FastConformer
    encoder + K-step graphed seq2seq decode over an EncoderDecoderCache)."""

    def __init__(self) -> None:
        super().__init__("starling", "cohere", supports_batch=False)

    def _load(self) -> None:
        from starling.cohere.pipeline import CohereMegaPipeline

        self.pipe = CohereMegaPipeline.from_pretrained()

    def _release(self) -> None:
        self.pipe = None

    def _run_one(self, audio: np.ndarray) -> str:
        texts, _ids = self.pipe.transcribe(audio, max_new_tokens=300)
        return texts[0]


class CohereStock(Engine):
    """Stock ``model.generate`` reference for cohere-transcribe.

    Seq2seq: passes the processor's ``input_features`` / ``attention_mask`` /
    ``decoder_input_ids`` through ``CohereAsrForConditionalGeneration.generate``
    (verified byte-exact against the manual eager reference in
    ``tests/test_cohere.py::test_reference_matches_generate``).
    """

    def __init__(self) -> None:
        super().__init__("stock transformers", "cohere", supports_batch=False)

    def _load(self) -> None:
        from starling.cohere.config import SAMPLE_RATE
        from starling.cohere.loader import load_model_and_processor

        self._sr = SAMPLE_RATE
        self.model, self.processor = load_model_and_processor()

    def _release(self) -> None:
        self.model = self.processor = None

    def _run_one(self, audio: np.ndarray) -> str:
        inp = self.processor(
            audio, sampling_rate=self._sr, language="en", return_tensors="pt"
        )
        feat = inp["input_features"].to(self.model.dtype).cuda()
        amask = inp["attention_mask"].cuda()
        dec_in = inp["decoder_input_ids"].cuda()
        prompt_len = dec_in.shape[1]
        gen = self.model.generate(
            input_features=feat, attention_mask=amask,
            decoder_input_ids=dec_in, max_length=prompt_len + 300,
        )
        gen_new = gen[:, prompt_len:]
        return self.processor.batch_decode(gen_new, skip_special_tokens=True)[0]


# ====================================================================== #
# higgs-audio-v3-stt  (runs under the isolated .venv-higgs, transformers 4.51)
# ====================================================================== #
class HiggsStarling(Engine):
    """starling fused pipeline for higgs-audio-v3-stt (Whisper-large-v3 mel
    encoder + MLP projector + Qwen3-1.7B decoder, graph-captured decode).

    NOTE: higgs-audio runs under its own isolated venv ``.venv-higgs``
    (``transformers==4.51.3``) because the model's ``trust_remote_code``
    modeling breaks under the repo's transformers 5.13. The package imports
    cleanly here, but ``_load`` only succeeds if launched with
    ``uv run --no-project --python .venv-higgs/bin/python``. See
    ``src/starling/higgs/UV_NOTES.md``. Gated by :func:`_higgs_keys` so the
    main-venv ``bench_all`` grid silently skips it.
    """

    def __init__(self) -> None:
        super().__init__("starling", "higgs", supports_batch=False)

    def _load(self) -> None:
        from starling.higgs.pipeline import HiggsMega

        self.pipe = HiggsMega.from_pretrained()

    def _release(self) -> None:
        self.pipe = None

    def _run_one(self, audio: np.ndarray) -> str:
        return self.pipe.transcribe(audio, sample_rate=16000, max_new_tokens=512)


class HiggsStock(Engine):
    """Stock reference for higgs-audio-v3-stt.

    Uses the vendored upstream ``transcribe()`` (``scripts/ref/transcribe.py``,
    run under ``.venv-higgs``) as the byte-exact oracle — the same path the
    golden reference is captured with. Like :class:`HiggsStarling`, this only
    loads under the isolated ``.venv-higgs``; gated by :func:`_higgs_keys`.
    """

    def __init__(self) -> None:
        super().__init__("stock transformers", "higgs", supports_batch=False)

    def _load(self) -> None:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "scripts" / "ref"))
        import transcribe as ref  # noqa: F401  (upstream transcribe())

        self._ref = ref
        from starling.higgs.loader import load_model_and_tokenizer

        self.model, self.tokenizer = load_model_and_tokenizer(attn_impl="eager")

    def _release(self) -> None:
        self.model = self.tokenizer = None

    def _run_one(self, audio: np.ndarray) -> str:
        wav = np.ascontiguousarray(audio, dtype=np.float32)
        return self._ref.transcribe(self.model, self.tokenizer, wav)


# ====================================================================== #
# CrispASR  (external ggml binary; granite + qwen3 backends only)
# ====================================================================== #
class CrispASR(Engine):
    """External CrispASR (ggml) binary subprocess. No batching; B loops on host.

    Backends: ``granite``, ``qwen3-1.7b``, ``parakeet``. The model slug passed
    to the base class is the bench model key (granite/qwen3/parakeet) so the
    adapter lands under the right column; the ggml ``--backend`` flag is the
    CrispASR-specific backend name.
    """

    def __init__(self, backend: str, gguf: str, model: str) -> None:
        super().__init__("CrispASR", model, supports_batch=False)
        self.backend = backend
        self.gguf = gguf
        self._wav_path: Optional[str] = None

    @property
    def available(self) -> bool:
        return CRISPASR_BIN.exists() and (CRISPASR_MODELS / self.gguf).exists()

    def _load(self) -> None:
        # No resident model; the binary loads it per-invocation. Write the
        # (varying) hypothesis audio to a temp wav the first time only.
        import tempfile

        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=str(REPO_ROOT)
        )
        self._tmp.close()
        self._wav_path = self._tmp.name

    def _release(self) -> None:
        if self._wav_path and os.path.exists(self._wav_path):
            try:
                os.unlink(self._wav_path)
            except OSError:
                pass
        self._wav_path = None

    def _run_one(self, audio: np.ndarray) -> str:
        import soundfile as sf

        sf.write(self._wav_path, audio, 16000, subtype="PCM_16")
        cmd = [
            str(CRISPASR_BIN),
            "--backend", self.backend,
            "-m", str(CRISPASR_MODELS / self.gguf),
            "-f", self._wav_path,
            "-n", "512",
            "--gpu-backend", "cuda",
            "-nt",  # no timestamps -> clean transcript lines
        ]
        p = subprocess.run(
            cmd, capture_output=True, text=True, env=_CRISPASR_ENV,
            cwd=str(ASR_BENCH), timeout=180,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"CrispASR failed (rc={p.returncode}):\n{p.stderr[-1000:]}"
            )
        # The transcript is the last non-empty stdout line (the binary prints
        # progress/timing to stderr with -nt; stdout holds just the text).
        lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
        return lines[-1] if lines else ""


class ParakeetCpp(Engine):
    """mudler's parakeet.cpp (C++/ggml) binary via the ld-linux shim wrapper.

    Reads the ``tdt-0.6b-v3-f16.gguf`` f16 model with the TDT decoder. Not
    batched; B loops on host (the binary is one-clip-per-process).
    """

    def __init__(self) -> None:
        super().__init__("parakeet.cpp", "parakeet", supports_batch=False)
        self._wav_path: Optional[str] = None

    @property
    def available(self) -> bool:
        return PARAKEET_CPP_WRAP.exists() and PARAKEET_CPP_MODEL.exists()

    def _load(self) -> None:
        import tempfile

        self._tmp = tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=str(REPO_ROOT)
        )
        self._tmp.close()
        self._wav_path = self._tmp.name

    def _release(self) -> None:
        if self._wav_path and os.path.exists(self._wav_path):
            try:
                os.unlink(self._wav_path)
            except OSError:
                pass
        self._wav_path = None

    def _run_one(self, audio: np.ndarray) -> str:
        import json as _json
        import soundfile as sf

        sf.write(self._wav_path, audio, 16000, subtype="PCM_16")
        cmd = [
            str(PARAKEET_CPP_WRAP), "transcribe",
            "--model", str(PARAKEET_CPP_MODEL),
            "--input", self._wav_path,
            "--decoder", "tdt",
            "--json",
        ]
        p = subprocess.run(
            cmd, capture_output=True, text=True, env=_PARAKEET_CPP_ENV,
            cwd=str(ASR_BENCH), timeout=180,
        )
        if p.returncode != 0:
            raise RuntimeError(
                f"parakeet.cpp failed (rc={p.returncode}):\n{p.stderr[-1000:]}"
            )
        # The CLI prints ggml init lines to stderr and one JSON object with a
        # "text" field on stdout. Tolerate leading/trailing non-JSON noise.
        blob = p.stdout
        i, j = blob.find("{"), blob.rfind("}")
        if i >= 0 and j > i:
            try:
                return _json.loads(blob[i:j + 1]).get("text", "").strip()
            except _json.JSONDecodeError:
                pass
        return blob.strip()


# ====================================================================== #
# Registry + filter resolution
# ====================================================================== #
def _qwen3_on_master() -> bool:
    try:
        import starling.qwen3  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _higgs_available() -> bool:
    """True iff the isolated ``.venv-higgs`` is present.

    ``starling.higgs`` imports cleanly in the main venv (heavy modelling is
    lazy), but its ``trust_remote_code`` modelling only *runs* under
    ``transformers==4.51`` (the isolated ``.venv-higgs``). Gate the engine keys
    on the venv existing so a main-venv ``bench_all`` sweep silently skips
    higgs instead of crashing inside ``_load``.
    """
    return (REPO_ROOT / ".venv-higgs" / "bin" / "python").exists()


# (engine_key, factory, requires_extra)
# starling/stock factories are cheap wrappers; CrispASR/Qwen3/Higgs are conditional.
ENGINE_REGISTRY: dict[str, Callable[[], Engine]] = {
    "starling-granite": GraniteStarling,
    "stock-granite": GraniteStock,
    "starling-parakeet": ParakeetStarling,
    "stock-parakeet": ParakeetStock,
    "starling-moss": MossStarling,
    "stock-moss": MossStock,
    "starling-ark": ArkStarling,
    "stock-ark": ArkStock,
    "starling-cohere": CohereStarling,
    "stock-cohere": CohereStock,
}


def _crispasr_keys() -> list[str]:
    keys = []
    if CrispASR("granite", "granite-speech-4.1-2b-f16.gguf", "granite").available:
        keys.append("crispasr-granite")
    if CrispASR("qwen3-1.7b", "qwen3-asr-1.7b-f16.gguf", "qwen3").available:
        keys.append("crispasr-qwen3")
    if CrispASR("parakeet", "cstr-parakeet-tdt-0.6b-v3-f16.gguf", "parakeet").available:
        keys.append("crispasr-parakeet")
    return keys


def _parakeet_cpp_keys() -> list[str]:
    if ParakeetCpp().available:
        return ["parakeet.cpp-parakeet"]
    return []


def _qwen3_keys() -> list[str]:
    if not _qwen3_on_master():
        return []
    return ["starling-qwen3", "stock-qwen3", "starling-batched-qwen3"]


def _higgs_keys() -> list[str]:
    if not _higgs_available():
        return []
    return ["starling-higgs", "stock-higgs"]


def available_keys() -> list[str]:
    """All engine keys usable in this checkout (qwen3/higgs/CrispASR/parakeet.cpp gated)."""
    return (list(ENGINE_REGISTRY) + _qwen3_keys() + _higgs_keys()
            + ["starling-batched-granite", "starling-spec-granite"]
            + _crispasr_keys() + _parakeet_cpp_keys())


def build_engines(
    models: list[str], engines: list[str]
) -> dict[str, list[Engine]]:
    """Resolve ``--models``/``--engines`` filters into ``{model: [engine, ...]}``.

    ``engines`` is a list of family names (``starling``, ``stock``,
    ``crispasr``); ``models`` is a list of model slugs. The cross product of
    available (engine, model) keys is built and instantiated (lazy: ``load()``
    is called later by the harness).
    """
    avail = available_keys()
    chosen: dict[str, list[Engine]] = {m: [] for m in models}
    for key in avail:
        # split off the trailing model slug (rsplit: family names like
        # "starling-batched" / "parakeet.cpp" can contain '-'/'.').
        fam, mdl = key.rsplit("-", 1)
        if mdl not in chosen:
            continue
        if engines and fam not in engines:
            continue
        if key.startswith("crispasr-"):
            backend_gguf = {
                "granite": ("granite", "granite-speech-4.1-2b-f16.gguf"),
                "qwen3": ("qwen3-1.7b", "qwen3-asr-1.7b-f16.gguf"),
                "parakeet": ("parakeet", "cstr-parakeet-tdt-0.6b-v3-f16.gguf"),
            }[mdl]
            chosen[mdl].append(CrispASR(backend_gguf[0], backend_gguf[1], mdl))
        elif key.startswith("parakeet.cpp-"):
            chosen[mdl].append(ParakeetCpp())
        elif key.startswith("starling-batched-"):
            # fam == "starling-batched"; mdl is the model slug
            chosen[mdl].append({"granite": GraniteStarlingBatched,
                                "qwen3": Qwen3StarlingBatched}[mdl]())
        elif key == "starling-spec-granite":
            chosen[mdl].append(GraniteStarlingSpec())
        elif key.startswith("qwen3") or mdl == "qwen3":
            cls = Qwen3Starling if fam == "starling" else Qwen3Stock
            chosen[mdl].append(cls())
        elif mdl == "higgs":
            cls = HiggsStarling if fam == "starling" else HiggsStock
            chosen[mdl].append(cls())
        else:
            chosen[mdl].append(ENGINE_REGISTRY[key]())
    return {m: es for m, es in chosen.items() if es}
