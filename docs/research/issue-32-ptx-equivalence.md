# Issue #32: PTX equivalence is not a byte-exact kernel gate yet

The proposed Workstream 2 gate cannot safely admit structural rewrites of
Starling's bf16 CUDA kernels today. Volta proves equivalence over mathematical
reals. Starling's contract is byte-exact bf16/fp32 behavior, including each
rounding point. These are different claims. This report records a concrete
false positive for the proposed gate and a bounded test on Starling's compiled
PTX. It does not add `z3-solver` or wire Volta into `bench_autotune.py`: neither
step would repair the missing floating-point semantics.

## Reproducible counterexample

[`scripts/verification/ptx_rounding_witness.cu`](../../scripts/verification/ptx_rounding_witness.cu)
contains two CUDA kernels. `staged_sum` adds bf16 operands with a bf16 result
after each addition. `direct_sum` adds the same operands in fp32, then rounds
once to bf16. With `x = 1` and `y = z = 1/256`, both compute the real sum
`1 + 1/128`. The first bf16 addition in `staged_sum` is a tie and rounds to
`1`; its second addition is the same tie. Its output is `0x3f80`. The single
final rounding in `direct_sum` produces `0x3f81`.

The [probe](../../scripts/verification/probe_ptx_rounding.py) compiles both
kernels to PTX with nvcc, executes them on the GPU, and compares the generated
PTX with Volta. On CUDA 13.0.88, sm_120, RTX 5090, and Volta commit
[`5d7530c`](https://github.com/willtunnels/volta/commit/5d7530cc7fbef656c3fbeac22c6529441e4db70c):

```text
$ taskset -c 0-15 python3 scripts/verification/probe_ptx_rounding.py \
    --volta /path/to/volta/target/release/volta
staged=0x3f80 direct=0x3f81
Exec: 0.000s  instructions: 34  block syncs: 0  warp syncs: 0
VC check: 0.000s (decision procedure 0.000s)  elements: 1
EQUIVALENT
byte_exact_gate=unsupported (real-arithmetic equivalence misses bf16 rounding)
```

The probe removes only `.ptr .align 1` parameter annotations from nvcc PTX
because this Volta parser rejects them. That normalization makes the experiment
possible; it does not make the comparison a proof about unmodified PTX. The
GPU byte result comes from running the original compiled CUDA kernels.

This result follows Volta's stated model, rather than exposing a bug in Volta.
Its [README](https://github.com/willtunnels/volta/blob/5d7530cc7fbef656c3fbeac22c6529441e4db70c/README.md#soundness-and-completeness)
says it treats floating-point values as reals. Its
[`add.bf16` lowering](https://github.com/willtunnels/volta/blob/5d7530cc7fbef656c3fbeac22c6529441e4db70c/crates/volta_analysis/src/lowering.rs#L1899-L1914)
discards the rounding mode, and its
[`cvt` lowering](https://github.com/willtunnels/volta/blob/5d7530cc7fbef656c3fbeac22c6529441e4db70c/crates/volta_analysis/src/lowering.rs#L3481-L3508)
treats float-to-float conversion as identity. NVIDIA's
[PTX ISA](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html)
defines bf16 operations and conversions with rounding behavior.

## Starling's compiled CUDA surface

The CUDA extension from issue #32 Workstream 1 PR #329, head
[`f32f18c`](https://github.com/sims1253/starling/commit/f32f18c592f59f8bdcff052b980d3f9444b6f107),
was compiled for sm_120. `cuobjdump --dump-ptx` produced a 154,904-byte file
with SHA-256 `05c55b7a1b0919aebd76554e1c0fc176c636a883866bab39ab99b0f797825e59`.
It contains 22 instantiated entries from the eight kernel families in
[`backend.cu`](../../src/starling/_kernels/cuda/backend.cu). The PTX contains
`add.bf16`, `mul.bf16`, `cvt.rn.bf16.f32`, `mul.ftz.f32`, and
`setp.neu.ftz.f32`. Some families also use approximate exponent, reciprocal,
or reciprocal square root instructions. An exact equivalence checker would
need the relevant rounding, FTZ, special-value, and approximation semantics.

Volta initially rejected the `cuobjdump` wrapper and then `.ptr .align 1` in
the first parameter declaration. Stripping the wrapper and that annotation
let it parse all 22 entries. A bounded `residual_add_kernel` analysis with
`N=1`, `alpha=1`, and one 32-thread CTA reported `z[0] = x[0] + y[0]` for the
actual `add.bf16` operation. Changing that operation in the extracted PTX to
`mul.bf16` produced `NOT EQUIVALENT: 1 mismatched element(s)`. This confirms
the parser and real-arithmetic checker can see a simple arithmetic mutant. It
does not validate bf16 bytes, other shapes, or the host launch contract.

To repeat the Starling-side probe, build the CUDA extension, extract PTX with
`cuobjdump --dump-ptx <extension.so>`, discard lines before the first
`.version`, remove `.ptr .align 1` from parameter declarations, then run
`volta parse` and `volta analyze` on a chosen entry. Preserve the original PTX
and its hash in any result record. Treat every normalization and every concrete
launch shape as a proof-scope limit.

## Decision and next boundary

Do not use a Volta `EQUIVALENT` verdict as the byte-exact autotune gate in
Workstream 2. A sound gate needs a source- or PTX-linked bit-vector or IEEE
floating-point model of the bf16/fp32 operations actually emitted, including
rounding and truncation, plus explicit launch shapes and aliasing assumptions.
Seeded moved-rounding, index, and tail mutants must be rejected before it can
admit a rewrite. The current sanitizer, byte-exact fixtures, and WER gates
remain the practical checks for candidate optimizations; they make finite
claims only. The proposed fused rewrite was not landed through an unsound
equivalence claim.

Primary sources: [issue #32](https://github.com/sims1253/starling/issues/32),
[Volta source at the tested commit](https://github.com/willtunnels/volta/tree/5d7530cc7fbef656c3fbeac22c6529441e4db70c),
[NVIDIA PTX ISA 9.0](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html).
