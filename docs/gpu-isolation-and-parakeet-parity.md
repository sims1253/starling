# GPU isolation (Task 1) + in-tree Parakeet parity gate (Task 2, Phase 0)

Implementation notes for the two selected tasks, including deviations from the
initial plan and the empirical findings.

## What changed versus the initial plan

The initial research used `windows-support` HEAD `5a43ffc`. By the time
this work ran, `origin/windows-support` had advanced to `3fb6510`, which is the
**merge of `ggml-perf` (PR #5)** and contains Task 2's headline code already:

* `cpp/parakeet/tdt_multistep.{cpp,hpp}` — the Parakeet K-step GPU decode
  (`d929bc6`) — **already merged** into `3fb6510`.
* `cpp/moss/*` whole-model graphs + K-step (`10861f4`) — **already merged**.

So the initial Task 2 step "cherry-pick `d929bc6`" is obsolete: that change
is upstream. This branch therefore **bases on `3fb6510`** (current upstream)
rather than re-doing merged work on the stale `5a43ffc` base. What Task 2 still
needed — and what this branch adds — is the **missing correctness seam**:

> The merge's `+37` lines to `tests/test_ggml_parity.py` added a *MOSS* K-step
> cache-boundary regression test, **not** the in-tree Parakeet parity gate. The
> only Parakeet test (`test_ggml_parakeet_byte_exact`) still drives the
> *external* parakeet.cpp server. The in-tree `libstarling_ggml` Parakeet engine
> (`StarlingGgmlParakeet`) had **no correctness gate at all**.

This branch adds that gate (Task 2, Phase 0). It lands test-first and is
validated end-to-end below.

## Task 1 — flock-based `GpuSession` + UUID keying + `starling-gpu-run`

Closes the two identified hazards:

1. **Orphaned native children** (`gpu_lock.py` recorded only the Python parent
   PID; native GPU subprocesses survived the parent and kept VRAM with no lock
   held). Mutual exclusion is now `fcntl.flock(LOCK_EX)` on a per-GPU flock
   file; the fd is opened **inheritable** (`os.set_inheritable(True)`, needed
   because PEP 446 makes Python fds non-inheritable by default) and passed to
   native children via `GpuSession.spawn(pass_fds=…)`. The lock is held for the
   child's whole lifetime even if the parent is SIGKILLed.
2. **`CUDA_VISIBLE_DEVICES` keying** (`CVD=0` vs unset vs `CVD=0,1` on one GPU
   produced different files). The key is now the **GPU UUID** actually visible
   (`nvidia-smi --query-gpu=uuid`), so every spelling of "this GPU" maps to one
   file. Multi-GPU visibility currently fails closed; correct support requires
   acquiring one lock per UUID in stable order so overlapping sets contend.

The old `starling.parakeet.gpu_lock` API (`with_gpu_lock` /
`acquire_gpu_lock` / `release_gpu_lock` / `GpuLockBusy`) is preserved as a thin
delegate so all existing call sites are unchanged; `GpuLockBusy` is the *same
class* as `starling.gpu.session.GpuLockBusy`. New additive module
`starling.gpu` (`session.py`, `run.py`); `starling-gpu-run` wraps any command
in the lock with zero per-script edits.

The flock implementation is POSIX-only. On native Windows it imports safely but
**fails closed** on acquisition; use an external machine-wide mutex or set
`STARLING_GPU_LOCK_DISABLE=1` only when benchmark serialization is guaranteed by
the caller.

Env knobs: `STARLING_GPU_LOCK_DISABLE`, `STARLING_GPU_LOCK_DIR`. The legacy
`STARLING_GPU_LOCK_FORCE` knob is accepted but deliberately does not kill a
holder: stale heartbeat metadata cannot distinguish a hung parent from a valid
native child that inherited the lock.

### Tests (`tests/test_gpu_session.py`, `tests/test_gpu_lock.py`) — CPU/fs only

The load-bearing one is `test_native_child_inherits_flock_fd`: a holder process
acquires + spawns `sleep`, is then SIGKILLed, and a second acquirer is **still
blocked** until the native child is killed — proving the fd-inheritance
invariant closes hazard #1. Plus: flock serialization, UUID collapse across CVD
variants, v2-token roundtrip (v1 rejected), the shim's unchanged signature, and
`starling-gpu-run` holding the lock for the exec'd child's lifetime. The 3
legacy `test_gpu_lock.py` cases were rewritten to assert the *equivalent
invariants under flock* (owner-safe release; live holders never stolen;
dead holders release immediately). All CPU — no GPU required to land.

## Task 2, Phase 0 — in-tree Parakeet parity gate

Adds `test_starling_ggml_parakeet_text_parity` (exact text) and
`test_starling_ggml_parakeet_idstream_parity` (exact **content-token** stream)
to `tests/test_ggml_parity.py`, plus the ctypes binding the id stream needs:
`GgmlModel.transcribe_pcm_ids` (`_native.py`) → the existing C symbol
`starling_ggml_parakeet_decode_ids_pub` (freed via `starling_ggml_free_string` —
the buffer is `std::malloc`'d and `free` is type-agnostic, so no new C entry
point is needed), and `StarlingGgmlParakeet._run_one_ids` (`engines.py`).

### Empirical finding — blanks differ, content tokens are exact

The initial plan assumed exact equality of the *full* id stream (incl. blanks).
Measured against the real `libstarling_ggml` + the `parakeet_tdt_*_ids.pt`
goldens (HF `model.generate().sequences[0]`):

| fixture | raw id len (ggml/golden) | raw first div | **non-blank equal?** |
|---------|--------------------------|---------------|----------------------|
| short   | 50 / 49                  | @18 (extra blank) | **yes** (44==44)  |
| medium  | 146 / 146                | none          | **yes** (132==132)   |
| long    | 473 / 472                | @17 (extra blank) | **yes** (437==437)|

The in-tree greedy loop and HF `model.generate` emit **blanks** (the TDT
"no-symbol-this-step" marker, id `blank_id`=8192) at slightly different
cadences; the **content** token sequence is byte-identical. Blanks are dropped
by detokenization (8192 is out of the piece range), so text matches anyway. The
id-stream gate therefore asserts exact equality of the **non-blank content
tokens** — stricter than text (token-level, pre-detokenization) and tolerant of
the known blank-cadence difference. `blank_id` is read from the golden `_meta.pt`
when present, else defaults to 8192.

### Validation (this step, under the Task-1 GPU lock)

Built lib was the `5a43ffc` serial-CPU greedy `.so` from the main checkout (the
K-step lives at `3fb6510` and is not this branch's change). With fixtures +
goldens present and the GPU lock held:

```
tests/test_ggml_parity.py::test_starling_ggml_parakeet_text_parity[short/medium/long]   PASSED
tests/test_ggml_parity.py::test_starling_ggml_parakeet_idstream_parity[short/medium/long] PASSED
6 passed
```

Reference timings for the serial engine (`STARLING_PARAKEET_TIMING=1`, warm):
decode ≈ 79 / 89 / 368 ms (short/medium/long); the ~9-10× K-step speedup is the
already-merged `3fb6510` code, not this branch.

### Reproducing

```bash
cmake -B build -DSTARLING_GGML_CUDA=ON -DSTARLING_GGML_SHARED=ON && cmake --build build -j
uv run python scripts/parakeet_tdt_golden.py        # generates golden/parakeet_tdt_*_ids.pt
export STARLING_GGML_PARAKEET_MODEL=…/tdt-0.6b-v3-f16.gguf
uv run pytest tests/test_ggml_parity.py -k parakeet  # skips cleanly if lib/model/goldens absent
```

The tests skip cleanly when `libstarling_ggml`, the model, or the goldens are
absent, so the pure-Python install keeps working unbuilt.

## Deferred (not this branch)

MOSS whole-model graphs + flash-attn decode are already merged at `3fb6510`, but
no extra work was done here. Vendor-don't-patch, energy/VAD segmentation, server
contracts, and speculative decoding remain investigations. Also note that
`docs/ggml-parakeet-perf-analysis.md` still cites external `parakeet.cpp` SHAs
as if they describe the in-tree implementation; that document needs a separate
accuracy cleanup.
