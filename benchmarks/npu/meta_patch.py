"""In-place rename of a .tflite metadata key (same length, no re-serialization).
python -I meta_patch.py IN OUT KEY [KEY...]   e.g. keep_stablehlo_constant"""
import sys

src, dst, keys = sys.argv[1], sys.argv[2], sys.argv[3:]
b = bytearray(open(src, "rb").read())
for k in keys:
    kb = k.encode()
    n = b.count(kb)
    assert n >= 1, (k, n)
    b = b.replace(kb, b"x" * len(kb))
    print(k, "renamed", n, "x")
open(dst, "wb").write(b)
