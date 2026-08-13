"""Generate the Hojo-ASR-V1 golden reference (``golden/hojo_reference.json``).

Runs the stock reference path (``HOJO_ASR.infer`` -> ``decoder_model.generate``
with beam-4) on the short/medium/long fixtures and records the emitted token
ids, transcript text, prompt length, and wall-clock seconds. The output gates
BOTH the Python CUDA megakernel and the C++/GGML engine for Hojo beam-4
decode, so it records the raw ``gen_ids`` stream (the winning beam's tokens,
exactly as ``generate`` returned them) -- not just the text -- because the
downstream ports need byte-exact id comparison.

Reference decode settings (from ``.hf-cache/hojo-asr-v1/config.yaml``):
    num_beams=4, do_sample=False, repetition_penalty=2.0, length_penalty=1,
    max_new_tokens=200 (clamped to ``min(200, feat_len*2+10)``),
    eos_token_id=151645 (``<|im_end|>``), pad token unused by generate.

``gen_ids`` here is the FULL return of the winning beam from
``decoder_model.generate(inputs_embeds=...)``. When ``generate`` is called with
``inputs_embeds`` (no ``input_ids``) it returns ONLY the newly generated tokens
(no prompt prefix), including the trailing eos (151645) when it stops on eos.
``text`` is derived from those ids via ``tokenizer.batch_decode(ids,
add_special_tokens=False)`` followed by the ``run_infer`` post-processing
(strip ``<|im_end|>`` / ``<|endoftext|>``, ``.strip()``), so detokenizing
``gen_ids`` reproduces ``text`` (the trailing eos detokenizes to empty after
the special-token strip).

The output is gitignored (it requires the 12 GB model); re-run this after
pulling a new model revision to refresh the reference.

Usage (from the repo root):
    .venv-hojo/bin/python scripts/make_hojo_golden.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import soundfile as sf
import torch

from hojo_asr import HOJO_ASR
from hojo_asr.dataset import infer_datapipe
from hojo_asr.utils import prepare_sample

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
MODEL_DIR = REPO_ROOT / ".hf-cache" / "hojo-asr-v1"
GOLDEN_PATH = REPO_ROOT / "golden" / "hojo_reference.json"

FIXTURE_NAMES = ("short", "medium", "long")


def _fixture_duration(path: Path) -> float:
    info = sf.info(str(path))
    return info.frames / info.samplerate


def infer_with_ids(
    model: HOJO_ASR, batch: Dict[str, Any], generation_config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Mirror ``HOJO_ASR.infer`` but also return ``gen_ids`` and ``prompt_len``.

    Returns one dict per batch element (in batch order) with keys:
        ``text_raw``  -- batch_decode(..., add_special_tokens=False)
        ``gen_ids``   -- winning beam token ids (int list), exactly as generate
                         returned them
        ``prompt_len``-- number of input embeds fed to the decoder
                         (1 bos + n_speech_embeds)
        ``n_speech_embeds`` -- speech embedding frame count
    """
    spectrogram = batch["spectrogram"]
    spectrogram_lengths = batch["spectrogram_lens"]
    batch_size = spectrogram.shape[0]

    speech_embeddings, speech_attn = model.encode_speech(
        spectrogram, spectrogram_lengths
    )

    bos_column = speech_attn[:, :1]
    bos_ids = (
        torch.ones(batch_size, 1, dtype=torch.int32, device=speech_embeddings.device)
        * model.bos_token_id
    )
    bos_embeds = model.decoder_model.model.embed_tokens(bos_ids)
    inputs_embeds = torch.cat([bos_embeds, speech_embeddings], dim=1)
    attention_mask = torch.cat([bos_column, speech_attn], dim=1)

    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnTokenSequences(StoppingCriteria):
        def __init__(self, stop_token_seqs=None):
            super().__init__()
            self.stop_token_seqs = stop_token_seqs or []

        def __call__(self, input_ids, scores, **kwargs):
            for seq in self.stop_token_seqs:
                tail = input_ids[0, -seq.numel():]
                if seq.numel() > 0 and torch.all(tail == seq).item():
                    return True
            return False

    stop_tensor = torch.tensor([-100], device=model.device)
    criteria = StoppingCriteriaList(
        [StopOnTokenSequences(stop_token_seqs=[stop_tensor])]
    )

    max_tokens = generation_config.get("max_new_tokens", 200)
    feat_len = speech_embeddings.size(1)
    max_new_tokens = min(max_tokens, int(feat_len * 2) + 10)
    max_new_tokens = max(max_new_tokens, 10)
    eos_token_id = model.tokenizer.eos_token_id

    output_ids = model.decoder_model.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        stopping_criteria=criteria,
        num_beams=generation_config.get("num_beams", 4),
        do_sample=generation_config.get("do_sample", False),
        min_length=generation_config.get("min_length", 1),
        temperature=generation_config.get("temperature", 1.0),
        top_p=generation_config.get("top_p", 0.9),
        repetition_penalty=generation_config.get("repetition_penalty", 1.0),
        length_penalty=generation_config.get("length_penalty", 1.0),
        attention_mask=attention_mask,
    )

    texts_raw = model.tokenizer.batch_decode(output_ids, add_special_tokens=False)
    results: List[Dict[str, Any]] = []
    for j in range(batch_size):
        results.append(
            {
                "text_raw": texts_raw[j],
                "gen_ids": output_ids[j].cpu().tolist(),
                "prompt_len": int(inputs_embeds.size(1)),
                "n_speech_embeds": int(speech_embeddings.size(1)),
            }
        )
    return results


def main() -> int:
    model = HOJO_ASR.load_model(str(MODEL_DIR), device="cuda:0")
    model.eval()
    generation_config = dict(model.config.generate)

    fixtures: Dict[str, Dict[str, Any]] = {}
    for fx in FIXTURE_NAMES:
        wav_path = FIXTURES / f"{fx}.wav"
        dur = _fixture_duration(wav_path)

        # Reuse the reference datapipeline so the mel extraction is byte-identical
        # to run_infer (single-utterance batch).
        datapipe, _ = infer_datapipe(
            [str(wav_path)], model.feat_extractor, batch_size=1
        )
        loader = torch.utils.data.DataLoader(datapipe, batch_size=None)

        t0 = time.perf_counter()
        per_batch: List[Dict[str, Any]] = []
        with torch.no_grad(), model.autocast_context():
            for batch in loader:
                batch = prepare_sample(batch, cuda_enabled=True, device=model.device)
                per_batch.append(
                    infer_with_ids(model, batch, generation_config)
                )
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0

        # Single utterance -> one element in one batch.
        rec = per_batch[0][0]
        text_raw = rec["text_raw"]
        text = (
            text_raw.replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
        gen_ids = rec["gen_ids"]
        prompt_len = rec["prompt_len"]

        # ---- consistency check: detokenize gen_ids the same way run_infer does
        detok_raw = model.tokenizer.decode(gen_ids, skip_special_tokens=False)
        detok = (
            detok_raw.replace("<|im_end|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
        if detok != text:
            raise SystemExit(
                f"[{fx}] gen_ids/text mismatch: "
                f"detok={detok!r} != text={text!r}"
            )

        fixtures[fx] = {
            "path": str(wav_path),
            "duration_s": round(dur, 6),
            "sample_rate": sf.info(str(wav_path)).samplerate,
            "text": text,
            "gen_ids": gen_ids,
            "gen_ids_len": len(gen_ids),
            "prompt_len": prompt_len,
            "n_speech_embeds": rec["n_speech_embeds"],
            "wall_s": wall_s,
            "decode": {
                "num_beams": generation_config.get("num_beams", 4),
                "do_sample": generation_config.get("do_sample", False),
                "repetition_penalty": generation_config.get(
                    "repetition_penalty", 2.0
                ),
                "length_penalty": generation_config.get("length_penalty", 1),
                "max_new_tokens": generation_config.get("max_new_tokens", 200),
                "min_length": generation_config.get("min_length", 1),
                "temperature": generation_config.get("temperature", 1.0),
                "top_p": generation_config.get("top_p", 0.9),
                "bos_token_id": int(model.bos_token_id),
                "eos_token_id": int(model.tokenizer.eos_token_id),
                "eos_token": "<|im_end|>",
                "stop_criteria": "StopOnTokenSequences([-100])",
            },
        }
        print(
            f"{fx}: dur={dur:.2f}s prompt_len={prompt_len} "
            f"gen_ids_len={len(gen_ids)} wall={wall_s:.2f}s "
            f"text={text[:80]!r}..."
        )

    golden = {
        "model": "HojoAI/Hojo-ASR-V1",
        "model_dir": str(MODEL_DIR),
        "path": "HOJO_ASR.infer -> decoder_model.generate(inputs_embeds=..., num_beams=4) (beam-4)",
        "fixtures": fixtures,
    }

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w") as f:
        json.dump(golden, f, indent=2)
    print(f"wrote {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
