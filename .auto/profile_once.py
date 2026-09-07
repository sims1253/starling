"""Stage profiling: mel vs encoder vs decode split, plus decode step counts."""
import ctypes, os, sys, time
import numpy as np

sys.path.insert(0, "/home/m0hawk/Documents/starling/tests/fixtures")
import make_fixtures as mkfx

REPO = "/home/m0hawk/Documents/starling"
MODEL = os.environ.get("PK_MODEL", f"{REPO}/models/parakeet-tdt-0.6b-v3-q8_0.gguf")

lib = ctypes.CDLL(os.environ["STARLING_GGML_LIB"])
lib.starling_ggml_load.restype = ctypes.c_void_p
lib.starling_ggml_load.argtypes = [ctypes.c_int, ctypes.c_char_p]
lib.starling_ggml_parakeet_mel_pub.restype = ctypes.POINTER(ctypes.c_float)
lib.starling_ggml_parakeet_mel_pub.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.POINTER(ctypes.c_int)]
lib.starling_ggml_parakeet_encode_pub.restype = ctypes.POINTER(ctypes.c_float)
lib.starling_ggml_encode_argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64, ctypes.POINTER(ctypes.c_int)]
lib.starling_ggml_parakeet_encode_pub.argtypes = lib.starling_ggml_encode_argtypes
lib.starling_ggml_parakeet_decode_pub.restype = ctypes.POINTER(ctypes.c_char)
lib.starling_ggml_parakeet_decode_pub.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
lib.starling_ggml_free_string.argtypes = [ctypes.POINTER(ctypes.c_char)]
lib.starling_ggml_free.argtypes = [ctypes.c_void_p]

ctx = lib.starling_ggml_load(1, MODEL.encode())
assert ctx
f32p = ctypes.POINTER(ctypes.c_float)
fx = mkfx.make_fixtures(write=False)

def profile(name):
    audio = np.ascontiguousarray(fx[name], dtype=np.float32)
    dur = audio.size / 16000
    p = audio.ctypes.data_as(f32p)
    outT = ctypes.c_int(0)
    # warmup each stage (graph build + replay cache)
    r = lib.starling_ggml_parakeet_encode_pub(ctx, p, audio.size, ctypes.byref(outT))
    assert r
    d = lib.starling_ggml_parakeet_decode_pub(ctx, p, audio.size)
    assert d
    mel_t = enc_t = dec_t = 0.0
    N = 1
    for _ in range(N):
        t0 = time.perf_counter(); r = lib.starling_ggml_parakeet_mel_pub(ctx, p, audio.size, ctypes.byref(outT)); t1 = time.perf_counter()
        mel_t += t1 - t0
        t0 = time.perf_counter(); r = lib.starling_ggml_parakeet_encode_pub(ctx, p, audio.size, ctypes.byref(outT)); t1 = time.perf_counter()
        enc_t += t1 - t0  # encode includes its own mel
        t0 = time.perf_counter(); d = lib.starling_ggml_parakeet_decode_pub(ctx, p, audio.size); t1 = time.perf_counter()
        dec_t += t1 - t0
        assert r and d
    mel_t /= N; enc_t /= N; dec_t /= N
    print(f"{name:7s} dur={dur:6.1f}s mel={mel_t*1000:7.1f}ms enc(incl mel)={enc_t*1000:7.1f}ms full_decode={dec_t*1000:7.1f}ms"
          f" | enc-only={(enc_t-mel_t)*1000:7.1f}ms dec-only={(dec_t-enc_t)*1000:7.1f}ms")

for n in ("medium",):
    profile(n)
lib.starling_ggml_free(ctx)
