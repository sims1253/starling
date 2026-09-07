"""Autoresearch benchmark: parakeet transcription speed + memory via ctypes.

Env knobs:
  STARLING_GGML_LIB   which libstarling_ggml.so to load (required)
  STARLING_GGML_DEVICE cpu | Vulkan0 (default: Vulkan0)
  PK_MODEL            GGUF path (default models/parakeet-tdt-0.6b-v3-q8_0.gguf)

Emits METRIC lines:
  METRIC rtf_vk=<medium-fixture RTF on selected device>   (primary when Vulkan)
  METRIC med_ms / short_ms / long_ms / peak_rss_mb
  METRIC sha1_12=<first 12 hex of transcript sha1> (output drift detector)
"""
import ctypes, hashlib, os, sys, time
import numpy as np

sys.path.insert(0, "/home/m0hawk/Documents/starling/tests/fixtures")
import make_fixtures as mkfx

REPO = "/home/m0hawk/Documents/starling"
MODEL = os.environ.get("PK_MODEL", f"{REPO}/models/parakeet-tdt-0.6b-v3-q8_0.gguf")
REPS = int(os.environ.get("PK_REPS", "5"))

lib = ctypes.CDLL(os.environ["STARLING_GGML_LIB"])
lib.starling_ggml_load.restype = ctypes.c_void_p
lib.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
lib.starling_ggml_transcribe_pcm.restype = ctypes.POINTER(ctypes.c_char)
lib.starling_ggml_transcribe_pcm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int]
lib.starling_ggml_free_string.argtypes = [ctypes.POINTER(ctypes.c_char)]
lib.starling_ggml_free.argtypes = [ctypes.c_void_p]

def read_rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return -1.0

fx = mkfx.make_fixtures(write=False)

ctx = lib.starling_ggml_load(1, MODEL.encode())
assert ctx, "load failed"

f32p = ctypes.POINTER(ctypes.c_float)
def transcribe(audio):
    ptr = lib.starling_ggml_transcribe_pcm(ctx, audio.ctypes.data_as(f32p), audio.size, 16000)
    assert ptr, "transcribe failed"
    txt = ctypes.cast(ptr, ctypes.c_char_p).value.decode("utf-8", "replace")
    lib.starling_ggml_free_string(ptr)
    return txt

def bench(name, reps):
    audio = np.ascontiguousarray(fx[name], dtype=np.float32)
    dur = audio.size / 16000
    out = transcribe(audio)  # warmup
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = transcribe(audio)
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times)//2]
    return med, dur, out

# Order: medium first (primary), then short, then long.
med_ms, med_dur, med_txt = bench("medium", REPS)
short_ms, short_dur, short_txt = bench("short", REPS)
long_ms, long_dur, long_txt = bench("long", max(2, REPS // 2))

h = hashlib.sha1((short_txt + "|" + med_txt + "|" + long_txt).encode()).hexdigest()[:12]
rss = read_rss_mb()
print(f"MED return-code ok")
print(f"METRIC rtf_med={med_ms/med_dur:.4f}")
print(f"METRIC med_ms={med_ms*1000:.1f}")
print(f"METRIC short_ms={short_ms*1000:.1f}")
print(f"METRIC long_ms={long_ms*1000:.1f}")
print(f"METRIC rtf_short={short_ms/short_dur:.4f}")
print(f"METRIC rtf_long={long_ms/long_dur:.4f}")
print(f"METRIC peak_rss_mb={rss:.0f}")
print(f"METRIC sha1_12={h}")
lib.starling_ggml_free(ctx)
