# Wave B — MOSS encoder/prefill perf (Phase 2e, encoder track)

## Why

After Wave A the MOSS decode is near peak parity (5.1-5.4 ms/tok vs 4.85), so
the remaining e2e gap vs the PyTorch reference is the **non-decode share**
(mel + audio encoder + adapter + prefill):

| length | ggml e2e | est. decode (tok x ms/tok) | est. non-decode | peak e2e | peak non-decode |
|--------|----------|-----------------------------|-----------------|----------|-----------------|
| short  | 323 ms   | 31 x 5.41 = 168 ms          | ~155 ms         | 184 ms   | ~35 ms          |
| medium | 725 ms   | 89 x ~5.3 = 470 ms          | ~255 ms         | 394 ms   | ~-              |
| long   | 1656 ms  | 187 x 5.14 = 960 ms         | ~695 ms         | 1433 ms  | ~525 ms         |

(gen/prompt token counts: 31/107, 89/300, 187/977.)

The non-decode cost is dominated by per-call graph BUILD + alloc, not compute:
- `encode_audio` (`cpp/moss/audio_encoder.cpp`) builds a fresh one-shot
  32-layer graph every call (`run_graph`), with a serial per-window attention
  loop + concat chain per layer that bloats node count on long audio.
- `apply_adapter` (`cpp/moss/adapter.cpp`) is another one-shot graph + an
  f32->bf16 host conversion of the whole encoder output between the two.
- `llm_prefill` (Wave A's `forward_prefill_new`) is a one-shot ~28-layer graph
  build per call.
- The PyTorch encoder is EAGER (25-40 ms on short) — a captured ggml encoder
  should beat it, exactly as parakeet's captured encoder beat the PyTorch one.

## Task

### B1. Persistent per-shape ReplayGraph for encoder+adapter

Fuse `encode_audio` + `apply_adapter` into ONE graph (the adapter is 3 linears
+ SiLU — the f32 round-trip between them is pure overhead; keep the exact same
cast boundaries *inside* the fused graph: encoder output f32 -> bf16 exactly as
the current host conversion does, i.e. `ggml_fp32_to_bf16` semantics =
`ggml_cast` to bf16). Cache it as a persistent `ReplayGraph` keyed on the mel
shape (key `(C, tail)` — chunk count + last-chunk frames fully determine every
shape in the graph), LRU-bounded (follow the parakeet encoder's `ReplayCache`
pattern in `cpp/parakeet/encoder.{hpp,cpp}`, including its bound). Inputs per
call: the packed `chunks` tensor + the `valid` index vector. Cache lives on the
moss context (`capi_moss.cpp`), cleared via `register_decode_cache_clearer`.

CPU backend: keep the one-shot path (capture is GPU-only; gate on
`Backend::is_gpu()` like everywhere else).

### B2. Batch the window loop

In each encoder layer the windowed attention currently loops
`for begin in 0..A step W` with per-window views + a concat chain. All windows
except possibly the tail have identical size S=W. Restructure to ONE batched
attention over `[D, W, H, n_full_windows]` (4D batched mul_mat) plus a single
tail-window pass when `A % W != 0`, writing results into the right rows
(preserving today's exact per-window math and output order — same kernels,
just batched). This cuts node count massively on long audio (32 layers x
n_windows sub-graphs -> 32 x 2) and speeds both build and GPU execution.
Gate: encoder output must stay identical to the current path (compare
max-abs-diff == 0 on the three fixtures via `STARLING_MOSS_DEBUG=1` style
capture or a small dev script), and the parity suite must stay green.

### B3. Per-shape ReplayGraph for prefill

Same treatment for `forward_prefill_new`: persistent ReplayGraph keyed on
prompt length, LRU-bounded, inputs = inputs_embeds (+ position/mask inputs as
in the decode design). The KV cache write path must keep working exactly as
Wave A left it (prefill fills the device cache the K-step graph reads).

### B4. Small wins (only if they fall out naturally)

- Skip the f32 host round-trip of `inputs_embeds` between prompt-merge and
  prefill if it is cheap to keep on device (prompt merge is host logic today —
  do NOT restructure prompt.cpp for this wave if it resists).
- Mel stays host-side (exactness) — do not touch.

## Requirements

Same hard rules as Waves A/C:
1. `uv run python -m pytest tests/test_ggml_parity.py -x -q` — 9/9, exit 0,
   after B1, B2, B3. Token-exact short/medium ids, transcript agreement long.
2. No edits under `third_party/ggml/`; no git state-changing commands.
3. CPU fallback keeps working; debug/dump env paths survive.
4. C API surface unchanged.

## Verification

```bash
cmake --build build -j
uv run python -m pytest tests/test_ggml_parity.py -x -q
uv run python benchmarks/bench_all.py --models moss \
  --engines starling,starling-ggml --lengths short,medium,long --batches 1 \
  --reps 10 --warmup 2
```

Success bar: short <= 230 ms, medium <= 560 ms, long <= 1250 ms, gates green.
Stretch: beat the PyTorch peak on short (< 184 ms) the way parakeet-ggml beat
its peak.

Report: files changed, per-stage timing if you instrument it (env-gated),
parity result, bench table.
