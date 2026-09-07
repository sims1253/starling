"""Quick probe: time parakeet transcription on CPU vs Vulkan via ctypes."""
import ctypes, os, sys, time
import numpy as np

sys.path.insert(0, "/home/m0hawk/Documents/starling/tests/fixtures")
import make_fixtures as mkfx

REPO = "/home/m0hawk/Documents/starling"
MODEL = os.environ.get("PK_MODEL", f"{REPO}/models/parakeet-tdt-0.6b-v3-q8_0.gguf")

lib = ctypes.CDLL(os.environ["STARLING_GGML_LIB"])
lib.starling_ggml_load.restype = ctypes.c_void_p
lib.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
lib.starling_ggml_transcribe_pcm.restype = ctypes.POINTER(ctypes.c_char)
lib.starling_ggml_transcribe_pcm.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.c_int]
lib.starling_ggml_free_string.argtypes = [ctypes.POINTER(ctypes.c_char)]
lib.starling_ggml_free.argtypes = [ctypes.c_void_p]

fx = mkfx.make_fixtures(write=False)
audio = np.ascontiguousarray(fx["medium"], dtype=np.float32)
n = audio.size
dur = n / 16000

ctx = lib.starling_ggml_load(1, MODEL.encode())
assert ctx, "load failed"
ptr = lib.starling_ggml_transcribe_pcm(ctx, audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, 16000)
text = ctypes.cast(ptr, ctypes.c_char_p).value.decode()
lib.starling_ggml_free_string(ptr)
print(f"warmup ({dur:.1f}s audio): {text[:80]!r}")

times = []
for i in range(5):
    t0 = time.perf_counter()
    ptr = lib.starling_ggml_transcribe_pcm(ctx, audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n, 16000)
    t1 = time.perf_counter()
    txt = ctypes.cast(ptr, ctypes.c_char_p).value.decode()
    lib.starling_ggml_free_string(ptr)
    times.append(t1 - t0)
    if i == 0:
        print(f"run0 text: {txt[:80]!r}")
best = min(times)
med = sorted(times)[len(times)//2]
print(f"median={med*1000:.1f}ms best={best*1000:.1f}ms rtf_med={med/dur:.3f} rtf_best={best/dur:.3f}")
