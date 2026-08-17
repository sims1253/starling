"""End-to-end MOSS megakernel pipeline test + bench vs stock golden."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    from starling.moss.pipeline import MossMegaPipeline

    print("[pipe] loading model + pipeline ...")
    t0 = time.perf_counter()
    pipe = MossMegaPipeline.from_pretrained(max_cache_len=1024, steps_per_replay=16)
    print(f"[pipe] built in {time.perf_counter()-t0:.1f}s")

    results = {}
    for name in ("short", "medium", "long"):
        wav, sr = sf.read(str(REPO / "tests" / "fixtures" / f"{name}.wav"))
        if wav.ndim > 1:
            wav = wav.mean(1)
        seconds = wav.shape[0] / sr
        inp = pipe.processor(wav.astype("float32"))
        inp = {
            k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inp.items()
        }

        # warmup (graph capture) then timed
        print(f"\n[pipe] {name}: {seconds:.2f}s warmup ...")
        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = pipe.transcribe(
                inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
                inp["audio_input_mask"], max_new_tokens=8,
            )
        torch.cuda.synchronize()
        print(f"[pipe] {name}: capture+warmup {time.perf_counter()-t0:.2f}s")

        # timed real run
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            text, ids = pipe.transcribe(
                inp["audio_data"], inp["audio_data_seqlens"], inp["input_ids"],
                inp["audio_input_mask"], max_new_tokens=200,
            )
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        rtfx = seconds / (ms / 1000.0)

        # correctness vs golden
        gold = torch.load(REPO / "golden" / f"moss_{name}_ids.pt")
        gold_text = (REPO / "golden" / f"moss_{name}_text.txt").read_text()
        tok_match = bool((ids[0] == gold[0]).all().item())
        text_match = text.strip() == gold_text.strip()
        print(f"[pipe] {name}: {ids.shape[1]} tokens, {ms:.0f}ms ({ms/ids.shape[1]:.2f}ms/tok), "
              f"RTFx {rtfx:.0f}x")
        print(f"[pipe]   token-exact={tok_match} text-exact={text_match}")
        print(f"[pipe]   {text[:120]}")
        results[name] = {
            "seconds": round(seconds, 2), "tokens": int(ids.shape[1]),
            "ms": round(ms, 1), "rtfx": round(rtfx, 1),
            "ms_per_tok": round(ms / ids.shape[1], 2),
            "token_exact": tok_match, "text_exact": text_match,
        }

    out = REPO / "outputs"
    out.mkdir(exist_ok=True)
    (out / "moss_bench.json").write_text(json.dumps(results, indent=2))
    print(f"\n[pipe] saved -> {out}/moss_bench.json")
    ok = all(r["token_exact"] for r in results.values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
