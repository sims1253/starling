"""Generate aarch64 forwarding trampolines for every OpenCL entry point the
vendor library exports, except the ones clshim.c wraps.

python3 -I gen_tramp.py VENDOR_libOpenCL.so clshim.c OUT.S
"""
import re
import subprocess
import sys

lib, csrc, out = sys.argv[1:4]
syms = subprocess.run(["nm", "-D", "--defined-only", lib], capture_output=True, text=True, check=True).stdout
names = sorted({m.group(1) for m in re.finditer(r"\s[TW]\s+(cl[A-Za-z0-9_]+)", syms)})
src = open(csrc).read()
wrapped = set(re.findall(r"^[a-z_][a-z_ ]*\*?\s*(cl[A-Z][A-Za-z0-9]*)\(", src, re.M))
wrapped |= set(re.findall(r"^cl_[a-z_]+\s+(cl[A-Z][A-Za-z0-9]*)\(", src, re.M))
wrapped |= set(re.findall(r"^void\*\s+(cl[A-Z][A-Za-z0-9]*)\(", src, re.M))
fwd = [n for n in names if n not in wrapped]
L = [".text"]
for i, n in enumerate(fwd):
    L += [f".globl {n}", f".type {n},%function", f"{n}:",
          f"  adrp x16, clshim_real+{8 * i}", f"  ldr x16, [x16, :lo12:clshim_real+{8 * i}]",
          "  cbz x16, 1f", "  br x16", "1: mov w0, #-59", "  ret"]
# Every name (forwarded or wrapped) gets a slot so clshim_init resolves them all.
allnames = fwd + sorted(wrapped)
L += [".bss", ".balign 8", ".globl clshim_real", ".hidden clshim_real", "clshim_real:",
      f"  .zero {8 * len(allnames)}",
      ".section .data.rel.ro,\"aw\"", ".balign 8", ".globl clshim_names", ".hidden clshim_names", "clshim_names:"]
L += [f"  .quad .Ln{i}" for i in range(len(allnames))]
L += [f".Ln{i}: .asciz \"{n}\"" for i, n in enumerate(allnames)]
L += [".balign 4", ".globl clshim_n", ".hidden clshim_n", "clshim_n:", f"  .word {len(allnames)}"]
open(out, "w").write("\n".join(L) + "\n")
print(f"{len(fwd)} forwarded, wrapped: {' '.join(sorted(wrapped))}")
