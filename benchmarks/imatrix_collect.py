"""Collect an activation importance matrix for calibrated quantization.

Runs the in-tree starling-ggml engine over calibration audio with
``STARLING_IMATRIX`` set: the collector hooks every MUL_MAT whose src[0] is a
named weight (cpp/runtime/imatrix.cpp, riding ggml_backend_sched's eval
callback) and the accumulated per-channel importance is flushed to disk at
process exit. Feed the result to starling-quantize's ``--imatrix``.

Collection forces observed nodes onto the CPU backend, so this is a slow,
offline pass — run it once per model family, not per serving process.

Usage::

    uv run python benchmarks/imatrix_collect.py \\
        --model models/parakeet-tdt-0.6b-v3-f32.gguf \\
        --output models/parakeet-tdt-0.6b-v3.imx.bin \\
        --tiers short,medium,long --repeats 2 [--wavs calib_dir/]

The fixtures alone are one speaker/one style; for real calibration mix in
``--wavs`` (16 kHz mono WAV/FLAC, any length) — a couple of hours of diverse
audio is plenty (parakeet v3 was trained on Granary, whose manifests are
CC-BY; pulling the audio requires the YODAS/MOSEL upstream corpora).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "fixtures"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="GGUF to collect with (f32 reference)")
    ap.add_argument("--output", required=True, help="imatrix output path (.bin)")
    ap.add_argument("--tiers", default="short,medium,long")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--wavs", default=None, help="optional dir of 16 kHz mono wavs/flacs")
    ap.add_argument("--mls-de", type=int, default=None, metavar="N",
                    help="additionally collect over N German clips from the "
                         "MLS de_de test split (multilingual calibration)")
    ap.add_argument("--fleurs", action="append", default=[], metavar="CFG:N",
                    help="collect over N clips of google/fleurs CFG (train "
                         "split), e.g. de_de:12; repeatable")
    ap.add_argument("--fleurs-25", type=int, default=None, metavar="N",
                    help="collect over N clips of EACH of parakeet-v3's 25 "
                         "languages (equal per-language budgets)")
    args = ap.parse_args()

    model = Path(args.model).expanduser()
    out = Path(args.output).expanduser()
    if not model.exists():
        print(f"error: model {model} does not exist")
        return 1

    # Must be set before the engine loads: the collector activates on first
    # graph build and flushes at process exit.
    os.environ["STARLING_IMATRIX"] = str(out.resolve())
    os.environ["STARLING_GGML_PARAKEET_MODEL"] = str(model.resolve())

    import make_fixtures as mkfx
    from engines import StarlingGgmlParakeet

    eng = StarlingGgmlParakeet()
    if not eng.available:
        print("error: engine unavailable (build/libstarling_ggml.so missing?)")
        return 1

    # Clips stream LAZILY into the transcription loop: a big --fleurs-25 run
    # otherwise materializes hundreds of clips (plus per-config streaming
    # buffers) before the model even loads, which both doubles peak memory
    # and hides all progress behind the download phase.
    def clips_iter():
        for _ in range(max(1, args.repeats)):
            for tier in tiers:
                yield f"fixture:{tier}", fixtures[tier]
        if args.wavs:
            for wav_path, wav_audio in wavs:
                yield f"wav:{wav_path.name}", wav_audio
        if mls_iter is not None:
            yield from mls_iter
        if fleurs_iter is not None:
            yield from fleurs_iter

    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    fixtures = mkfx.load_fixtures()

    wavs: list = []
    if args.wavs:
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            print("error: --wavs needs numpy+soundfile")
            return 1
        wdir = Path(args.wavs).expanduser()
        for p in sorted(list(wdir.glob("*.wav")) + list(wdir.glob("*.flac"))):
            audio, sr = sf.read(p, dtype="float32", always_2d=False)
            if sr != 16000:
                print(f"[skip] {p.name}: {sr} Hz != 16000 (resample first)")
                continue
            wavs.append((p, np.ascontiguousarray(audio)))

    mls_iter = None
    if args.mls_de:
        import numpy as np
        from datasets import load_dataset
        ds = load_dataset("facebook/multilingual_librispeech", "german",
                          split="test", streaming=True)

        def _mls():
            got = 0
            for i, ex in enumerate(ds):
                if got >= args.mls_de:
                    return
                aud = ex["audio"]
                if hasattr(aud, "get_all_samples"):  # datasets>=5 lazy decoder
                    s = aud.get_all_samples()
                    array, sr = s.data.numpy(), int(s.sample_rate)  # [ch, n]
                else:
                    array, sr = aud["array"], int(aud["sampling_rate"])
                array = np.asarray(array, dtype=np.float32)
                if array.ndim == 2:  # mono arrives as [1, n]
                    array = array[0] if array.shape[0] in (1, 2) else array.reshape(-1)
                if sr != 16000:  # MLS opus decodes at 48 kHz
                    import torch
                    import torchaudio.functional as AF
                    array = AF.resample(torch.from_numpy(array), sr, 16000).numpy()
                    sr = 16000
                assert sr == 16000, f"MLS clip at {sr} Hz"
                yield f"mls-de:{i}", np.ascontiguousarray(array)
                got += 1
        mls_iter = _mls()

    fleurs_iter = None
    if args.fleurs or args.fleurs_25 is not None:
        from fleurs_util import PARAKEET_V3_LANGS, fleurs_clips
        budget: dict[str, int] = {}
        for spec in args.fleurs:
            cfg, _, n = spec.partition(":")
            budget[cfg] = int(n)
        if args.fleurs_25 is not None:
            for cfg in PARAKEET_V3_LANGS:
                budget[cfg] = max(budget.get(cfg, 0), args.fleurs_25)
        print(f"[fleurs] streaming {sum(budget.values())} clips across "
              f"{len(budget)} configs", flush=True)
        fleurs_iter = ((label, audio)
                       for label, audio, _text in fleurs_clips(budget, split="train"))

    print(f"collecting -> {out}", flush=True)
    eng.load()
    for i, (label, audio) in enumerate(clips_iter()):
        text = eng.transcribe(audio)[0]
        print(f"[{i + 1}] {label}: {len(audio) / 16000:.1f}s "
              f"-> {len(text)} chars", flush=True)
    # Flush the collector through the C API BEFORE any teardown so the file
    # exists regardless of exit-time behavior, then skip the (crash-prone)
    # interpreter teardown entirely.
    from starling._ggml import _native
    lib = getattr(_native, "_LIB", None) or _native._load_lib()
    flush = getattr(lib, "starling_ggml_imatrix_flush_pub", None)
    if flush is not None:
        flush()
    eng.close()
    print(f"done; imatrix flushed to {out}")
    os._exit(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
