"""Clean uncontended full-pipeline benchmark (GPU lock held, multiple iters).

Resolves the earlier measurement ambiguity (532ms warm vs 1082ms contended).
Reports median + min for stock / non-spec mega / spec mega, RTFx, tok/s.
"""
import sys
import statistics
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))  # noqa: E402

import torch  # noqa: E402
from starling.parakeet.gpu_lock import with_gpu_lock  # noqa: E402
from starling.granite.audio import build_inputs, load_sample_audio  # noqa: E402
from starling.granite.loader import load_model_and_processor  # noqa: E402
from starling.granite.pipeline import MegaPipeline  # noqa: E402

def wall_ms(fn, warm=3, iters=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter()-t0)*1000)
    return statistics.median(ts), min(ts)

with with_gpu_lock(session="granite", model="granite-speech-4.1-2b",
                   eta_min=5, note="clean uncontended baseline"):
    print("loading...", flush=True)
    model, proc = load_model_and_processor("eager")
    wav, sr = load_sample_audio()
    inputs = build_inputs(proc, wav)
    feats = inputs["input_features"].bfloat16()
    ids = inputs["input_ids"]
    mask = inputs.get("input_features_mask")
    n_tok = ids.shape[1]
    dur = wav.shape[1]/sr
    pipe = MegaPipeline(model, proc, encoder_mode="cudagraph", use_fused_llm=True)
    print(f"audio {dur:.1f}s, prompt {n_tok} tokens\n", flush=True)

    print(f"{'path':<28}{'median ms':>12}{'min ms':>12}{'tok/s':>10}{'RTFx':>10}{'vs stock':>11}")
    print("-"*83)

    # stock transformers generate
    def stock():
        with torch.inference_mode():
            model.generate(input_ids=ids, input_features=feats,
                           attention_mask=inputs["attention_mask"],
                           input_features_mask=mask, max_new_tokens=100,
                           do_sample=False, num_beams=1)
    smed, smin = wall_ms(stock, warm=2, iters=4)
    print(f"{'stock transformers':<28}{smed:>12.1f}{smin:>12.1f}{'':>10}{'':>10}{'1.00x':>11}")

    # mega non-spec
    def nonspec():
        pipe.transcribe(feats, ids, input_features_mask=mask, max_new_tokens=100, speculative=False)
    nmed, nmin = wall_ms(nonspec)
    ntok = 100
    print(f"{'mega (non-spec)':<28}{nmed:>12.1f}{nmin:>12.1f}{ntok/(nmed/1000):>10.1f}{dur/(nmed/1000):>10.2f}x{smed/nmed:>11.2f}x")

    # mega spec
    def spec():
        pipe.transcribe(feats, ids, input_features_mask=mask, max_new_tokens=100, speculative=True)
    emed, emin = wall_ms(spec)
    print(f"{'mega (speculative)':<28}{emed:>12.1f}{emin:>12.1f}{ntok/(emed/1000):>10.1f}{dur/(emed/1000):>10.2f}x{smed/emed:>11.2f}x")
    print("-"*83)
    print(f"spec vs non-spec speedup: {nmed/emed:.2f}x")
