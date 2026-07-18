# Starling Future Work

This file now contains only unresolved ideas worth investigating after the
correctness, server, benchmark, dependency, documentation, and MOSS FP8 fixes in
`a3b8072` and `1cfb165`. These are opportunities or conditional architecture
work, not known release-blocking defects.

## High-value performance experiments

### 1. Incremental streaming inference

The `/stream` path still transcribes overlapping fixed windows independently.
Explore caching encoder output for the committed prefix, or carrying decoder KV
state forward and extending it with new audio. This removes whole encoder and
prefill passes and should improve perceived dictation latency more than another
decode-step micro-optimization.

Questions to answer:

- Can prefix state survive overlap correction without changing committed text?
- How should reset/commit/cancellation invalidate cached state?
- What is the latency and VRAM delta on a long dictation trace?

### 2. Parakeet as a cross-model speculative draft

Parakeet-TDT is fast enough to draft text for moss, qwen3, ark, or higgs at
relatively low incremental cost. Prototype drafting with Parakeet, retokenizing
into the verifier vocabulary, and verifying several tokens in one graphed
multistep replay.

Measure acceptance rate, end-to-end speedup, WER drift, and the roughly 1.2 GB
extra VRAM cost. This is promising because it reduces sequential decode steps,
the bottleneck that launch-level optimizations do not remove.

### 3. Extend the fused FP8 GEMV to other decoders

Granite and MOSS now share the graph-safe fused Triton dequant-GEMV for M=1
decode. Qwen3, ark, and higgs expose similar fused decoder seams, but each still
needs model-specific quantized-weight wiring and independent end-to-end latency,
WER, and transcript-drift validation before enabling FP8.

## Conditional architecture work

### 4. Request coalescing for shared deployments

The server intentionally serializes inference, which is appropriate for a
single-user sidecar. If it becomes a shared multi-client endpoint, prototype
coalescing compatible queued requests into the existing Granite/Qwen3 batched
pipelines. Compare throughput, tail latency, batching delay, and VRAM before
committing to the added scheduler complexity.

Do not substitute concurrent replay of the same captured CUDA graphs: their
static buffers make that unsafe without duplicating pipeline state per stream.

### 5. Shared graph-cache and replay-policy helper

The current model graph caches are bounded, so there is no active unbounded-OOM
bug. The bounded cache mechanics and model-specific K-selection policies are
still duplicated, however. A future ninth model would benefit from a small
shared helper providing:

- bounded LRU ownership and graph cleanup;
- shape bucketing/key normalization;
- explicit model-specific replay-step selection;
- cache metrics for captures, hits, misses, and evictions.

Keep the thresholds model-specific; centralize lifecycle safety rather than
forcing one tuning policy across incompatible decoders.

### 6. Finish transport consolidation

HTTP transcription and error mapping are now shared, but FastAPI and the stdlib
fallback still have separate request parsing and WebSocket protocol loops.
Either extract a transport-neutral stream command/session handler or confirm
FastAPI is always available and remove the hand-written RFC 6455 fallback.

### 7. Multi-process fairness and public deployment hardening

In-process requests are FIFO and all worktrees now share an ownership-safe GPU
lock. Multiple independent server/benchmark processes can still race for the
next lock acquisition. Add a cross-process ticket queue only if that deployment
becomes real.

Likewise, the server warns on non-loopback binds but deliberately has no auth.
If public binding becomes supported, put authentication, TLS termination, and
rate limiting in front of the service rather than growing them ad hoc inside
the model server.

## Release cleanup

Before a public release, revisit the de-internalization checklist in
`comms.md`: remove stale agent-handoff prose and scratch comments from tracked
files, then refresh or retire the stale internal notes.

## Deprioritized directions

- `nvfp4_weights` needs quantized weights plus a QAD fine-tune and targets
  bandwidth even though current decode is primarily launch-bound. Revisit only
  alongside a technique that also reduces launch count.
- More isolated kernel work should rank below incremental streaming,
  speculative verification, and shipping validated batching where workload
  measurements show those larger seams matter.
