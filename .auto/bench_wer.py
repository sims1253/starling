"""Quality gate: WER of the parakeet engine on the real LibriSpeech corpus
(32 cached utterances) + the three deterministic fixtures.

Env: STARLING_GGML_LIB, STARLING_GGML_DEVICE, PK_MODEL (same as bench_speed).
Prints:
  METRIC wer_real=<mean per-clip WER %% on real corpus>
  METRIC wer_fix=<mean WER %% on short/medium/long fixtures>
  METRIC cer_real=<mean per-clip CER %%>
  VERDICT ok|fail  (fail if wer_real exceeds baseline + tolerance)
"""
import ctypes, json, os, sys
import numpy as np

sys.path.insert(0, "/home/m0hawk/Documents/starling/tests/fixtures")
sys.path.insert(0, "/home/m0hawk/Documents/starling/benchmarks")
import make_fixtures as mkfx
import get_real_corpus as grc
from wer import REFERENCE_TRANSCRIPTS, wer_pct, cer_pct

REPO = "/home/m0hawk/Documents/starling"
MODEL = os.environ.get("PK_MODEL", f"{REPO}/models/parakeet-tdt-0.6b-v3-q8_0.gguf")
# Baseline (q8_0) real-corpus WER captured at session start; tolerance from
# the noise floor of repeated evals (+1.0 abs points). See .auto/quality.json.
BASELINE = json.load(open(f"{REPO}/.auto/quality.json"))
TOL = float(os.environ.get("PK_WER_TOL", "1.0"))

lib = ctypes.CDLL(os.environ["STARLING_GGML_LIB"])
lib.starling_ggml_load.restype = ctypes.c_void_p
lib.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
lib.starling_ggml_transcribe_pcm.restype = ctypes.POINTER(ctypes.c_char)
lib.starling_ggml_transcribe_pcm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int]
lib.starling_ggml_free_string.argtypes = [ctypes.POINTER(ctypes.c_char)]
lib.starling_ggml_free.argtypes = [ctypes.c_void_p]

ctx = lib.starling_ggml_load(1, MODEL.encode())
assert ctx, "load failed"
f32p = ctypes.POINTER(ctypes.c_float)

def transcribe(audio):
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    ptr = lib.starling_ggml_transcribe_pcm(ctx, audio.ctypes.data_as(f32p), audio.size, 16000)
    assert ptr, "transcribe failed"
    txt = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", "replace")
    lib.starling_ggml_free_string(ptr)
    return txt.strip()

# Real corpus: per-clip WER (the eval.json convention: mean per-clip WER).
corpus = grc.load_real_corpus(n=32)
wers, cers = [], []
for audio, sr, ref in corpus:
    hyp = transcribe(audio)
    wers.append(wer_pct(ref, hyp))
    cers.append(cer_pct(ref, hyp))
wer_real = float(np.mean(wers))
cer_real = float(np.mean(cers))

# Fixtures (repeat-bias, gross-breakage check).
fx = mkfx.make_fixtures(write=False)
fx_wers = [wer_pct(REFERENCE_TRANSCRIPTS[n], transcribe(fx[n])) for n in ("short", "medium", "long")]
wer_fix = float(np.mean(fx_wers))

print(f"METRIC wer_real={wer_real:.2f}")
print(f"METRIC cer_real={cer_real:.2f}")
print(f"METRIC wer_fix={wer_fix:.2f}")

base = BASELINE["wer_real_baseline"]
limit = base + TOL
if wer_real > limit:
    print(f"VERDICT fail (wer_real {wer_real:.2f} > baseline {base:.2f} + {TOL})")
    sys.exit(1)
print("VERDICT ok")
lib.starling_ggml_free(ctx)
