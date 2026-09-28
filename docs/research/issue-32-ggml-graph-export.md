# Issue #32: ggml graph extraction boundary

Workstream 3 can start with a real graph export. It cannot yet make an
all-input equivalence claim. The new `STARLING_GRAPH_EXPORT_DIR` switch writes
JSON snapshots for both one-shot `Backend::compute` graphs and captured
`ReplayGraph` graphs after ggml builds them. It is off by default. The
[snapshot code](../../cpp/runtime/graph_snapshot.cpp) records tensor IDs,
operation names, dimensions, byte strides, all source slots, view source and
offset, raw operation parameters, flags, output, captures, and side-effect
roots. It records the selected device and ggml commit. It never reads model
weights or input tensor values.
The full 64-byte operation-parameter field and all source slots are stable
for the pinned ggml revision because `ggml_new_tensor_impl` initializes the
whole tensor, including those arrays, to zero before an op fills them. Recheck
that behavior when updating the submodule.

Use a fresh directory for an engine run, then inspect the result:

```bash
cmake -S . -B build-issue32 -DSTARLING_GGML_CUDA=OFF -DSTARLING_GGML_TESTS=ON
cmake --build build-issue32 --target granite_trace_test -j 4
STARLING_GRAPH_EXPORT_DIR=/tmp/starling-graphs build-issue32/granite_trace_test
python3 scripts/verification/inspect_ggml_graphs.py /tmp/starling-graphs
```

On the synthesized tiny Granite model, the native test passed and produced
116 valid snapshots. The largest graph had 345 executed nodes and 417 tensors;
the files totaled 6,106,731 bytes. The inspector checked every tensor ID,
source reference, view reference, output, capture, side-effect root, four-axis
shape and stride, and 64-byte operation-parameter encoding. The generated
graphs included `MUL_MAT`, `RMS_NORM`, `SOFT_MAX`, `VIEW`, `CPY`, and other real
ggml operations. A direct CPU `ReplayGraph` in the same test also exported its
two-tensor `ADD` graph. This exercises the actual Starling graph-build paths;
it does not establish a result for a full Higgs or Hojo checkpoint.

The exporter uses an iterative walk. A first recursive version crashed on
the real one-shot graph chain; the test above now passes. Each snapshot is
written to a temporary file and renamed only after a complete write. The
switch creates one file for each graph build until the per-process cap, so a
long one-shot decode run can produce many files. With the switch absent, each
graph build only checks the environment variable.

## Why the verdict is still unknown

The snapshot explicitly says `semantics=not_encoded` and
`leaf_values=not_exported`. Two graphs can have identical topology and names
while using different weights. A `CPY` root can change persistent state even
when the returned output tensor is the same. `MUL_MAT` and quantized operations
also depend on the selected backend and its numeric implementation. These
facts block the proposed comparison of output expressions alone. The
[inspector](../../scripts/verification/inspect_ggml_graphs.py) validates
structure and always prints `equivalence=unknown`.

Export errors from an explicitly enabled directory fail the graph build, so
an incomplete snapshot cannot be mistaken for a successful verification run.
At most 256 snapshots are written per process; the exporter warns once when
the limit is reached. Run a shorter workload or a separate process if later
graph shapes are needed. Snapshots from concurrent processes include their
process IDs in filenames.

The fixed-width `op_params_hex` field and the full `src` slot walk are
deterministic only because the pinned ggml's `ggml_new_tensor_impl`
zero-initializes the whole tensor struct. Recheck that after any ggml bump: a
change there would add noise to every snapshot diff rather than fail loudly.

The export follows the public ggml tensor fields and graph accessors:
[`ggml_tensor` and `ggml_graph_node`](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/include/ggml.h).
Starling's [one-shot build](../../cpp/runtime/backend.cpp) and replay build
already expose the cgraph after all output, capture, and side-effect roots
have been expanded. ggml also exposes a DOT dump, but it does not provide the
complete typed/strided operation record needed for translation validation.

The next defensible checker slice is a stateless subgraph with explicit input
and constant bytes and a small set of operations whose **backend-specific**
index and rounding semantics have been encoded. `VIEW`, `RESHAPE`, and
`PERMUTE` are candidates for an exact index-map proof before reductions or
quantized matrix multiplication. A checker must reject unsupported ops,
unknown weights, mutable roots, timeout, and changed device semantics as
`unknown`; only then should a seeded wrong stride or moved rounding point be
used to test its rejection path. No graph optimization or autotune gate was
landed on the basis of a topology snapshot.

Primary sources: [issue #32](https://github.com/sims1253/starling/issues/32),
[pinned ggml API](https://github.com/ggml-org/ggml/blob/e91ded11bdcd78c42f9c8d3978ff6686eb4c1226/include/ggml.h),
[Starling ReplayGraph](../../cpp/runtime/backend.cpp).
