"""RTF benchmark for higgs-audio-v3-stt: starling (mega) vs stock transformers.

Runs both paths on short/medium/long fixtures, under the GPU lock (so timings
are uncontended), and prints an RTFx table. ``stock`` = the model's own
``model.generate(do_sample=False)`` (the byte-exact reference); ``starling`` =
the CUDA-graph-captured decode (single- or multi-step).

Run under the isolated venv via uv (see src/starling/higgs/UV_NOTES.md):
    uv run --no-project --python .venv-higgs/bin/python python benchmarks/bench_higgs_rtf.py [--decoder single|multi]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(
    os.environ.get("STARLING_FIXTURES_DIR", REPO / "tests" / "fixtures")
).expanduser()
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "ref"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import soundfile as sf  # noqa: E402

from starling.higgs.config import EOS_TOKEN_IDS  # noqa: E402
from starling.higgs.loader import load_model_and_tokenizer, make_collator  # noqa: E402
from starling.higgs.llm_mega import LLMMega  # noqa: E402
from starling.higgs.multistep import MultiStepLLMMega  # noqa: E402
import transcribe as ref  # noqa: E402  # upstream reference transcribe()

FIXTURES = [
    ("short", FIXTURES_DIR / "short.wav"),
    ("medium", FIXTURES_DIR / "medium.wav"),
    ("long", FIXTURES_DIR / "long.wav"),
]


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    a, sr = sf.read(str(path))
    if a.ndim > 1:
        a = a.mean(axis=1)
    return np.asarray(a, dtype=np.float32), sr


def _build_batch(collator, tok, audio_np):
    input_ids = ref._build_input_tokens(tok, ref.DEFAULT_PROMPT, enable_thinking=True)
    sample = ref._build_sample(audio_np, input_ids, sample_rate=16000)
    batch = asdict(collator([sample]))
    return {k: (v.to("cuda").contiguous() if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def stock_transcribe(model, tok, audio_np, sr, max_new_tokens):
    """Stock transformers reference: model.generate() (byte-exact golden)."""
    t0 = time.perf_counter()
    text = ref.transcribe(model, tok, audio_np, sample_rate=sr, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize()
    return text, (time.perf_counter() - t0) * 1000.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default="multi", choices=["single", "multi"])
    ap.add_argument("--k", type=int, default=2, help="steps_per_replay for multi")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=1, help="repeats for starling (best-of)")
    args = ap.parse_args()

    model, tok = load_model_and_tokenizer()
    collator = make_collator(model)

    print(f"decoder={args.decoder} K={args.k}  (RTFx = audio_s / transcribe_s, higher=faster)\n")
    print(f"{'audio':6} {'dur':>5} {'stock_ms':>10} {'stock_rtfx':>10} {'star_ms':>9} {'star_rtfx':>10} {'speedup':>9}")
    results = {}
    for name, path in FIXTURES:
        audio_np, sr = _load_audio(path)
        dur = len(audio_np) / sr
        # cap new tokens per fixture (avoid runaway on tiled audio)
        mnt = min(args.max_new_tokens, 160 if name == "short" else 300)

        # stock reference (one run; it's the golden)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            s_text, stock_ms = stock_transcribe(model, tok, audio_np, sr, mnt)

        # starling (best of repeats)
        batch = _build_batch(collator, tok, audio_np)
        best_ms = float("inf")
        best_ids = None
        for _ in range(max(args.repeats, 1)):
            # fresh decoder each repeat so capture + cache are clean
            if args.decoder == "multi":
                llm = MultiStepLLMMega(model, max_cache_len=2048, steps_per_replay=args.k)
            else:
                llm = LLMMega(model, max_cache_len=2048)
            r = llm.generate(batch, max_new_tokens=mnt, eos_token_ids=EOS_TOKEN_IDS, tokenizer=tok)
            if r.total_ms < best_ms:
                best_ms = r.total_ms
                best_ids = r.ids[0].tolist()
        star_text = ref._parse_output(
            tok.decode(torch.tensor(batch["input_ids"][0].tolist() + best_ids), skip_special_tokens=False)
        )

        stock_rtfx = dur / (stock_ms / 1000.0)
        star_rtfx = dur / (best_ms / 1000.0)
        speedup = stock_ms / best_ms
        print(f"{name:6} {dur:5.1f} {stock_ms:10.0f} {stock_rtfx:10.0f}x {best_ms:9.0f} {star_rtfx:10.0f}x {speedup:9.2f}x")
        results[name] = {
            "duration_s": dur, "stock_ms": stock_ms, "star_ms": best_ms,
            "stock_rtfx": stock_rtfx, "star_rtfx": star_rtfx, "speedup": speedup,
            "stock_text": s_text, "star_text": star_text,
            "match": s_text.strip() == star_text.strip(),
        }

    out = {"decoder": args.decoder, "K": args.k, "fixtures": results}
    outp = REPO / "outputs" / "higgs_bench.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
