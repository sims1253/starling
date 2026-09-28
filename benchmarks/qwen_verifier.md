# Native Qwen batched verifier (#311)

The opt-in `speculative_generate` API accepts a proposer callback and a
maximum draft length. It verifies `[last_verified_token, draft...]` in one
causal decoder graph. On a mismatch, it emits the target token and rewinds the
logical KV length so the next pass overwrites the rejected suffix. It handles
EOS, the output budget, the cache limit, and cancellation before committing
tentative output. MOSS, Granite, Qwen3-ASR, and S1 expose the shared API.
The normal greedy path is unchanged.

The CPU CI test builds a small Granite GGUF with a changing token sequence.
It checks full acceptance, rejection at each of four draft positions, EOS
inside a draft, the exact cache boundary, an invalid boundary, an empty-draft
fallback, and cancellation after the verify graph. The same test can compare
a real MOSS GGUF with saved input embeddings:

```sh
cmake -S backends/native -B /tmp/starling-311-build -DSTARLING_GGML_SHARED=ON
cmake --build /tmp/starling-311-build --target speculative_verifier_test
STARLING_GGML_DEVICE=cpu /tmp/starling-311-build/speculative_verifier_test \
  /path/to/moss-transcribe-preview-2b-bf16-exact.gguf \
  /path/to/moss_short_inputs_embeds.f32
```

The local real-model run used a MOSS GGUF with SHA-256
`b96dae2bcadc9e89f61a3ac1e103915e0abb11da521ca6d8db91e3244129e71a`
and 107-token saved short golden input embeddings with SHA-256
`7fb06ee9ca909e953d67c00d00b9ad30b61bb5be2dd4eaf9fb1ce8535e75c143`.
After one untimed greedy warmup, the 12-token CPU runs were:

| Run | Output match | Accepted/proposed | Total wall time |
| --- | --- | ---: | ---: |
| Target-only greedy | Baseline | — | 2419.8 ms |
| Perfect oracle draft, K=4 | Exact | 8/8 | 1562.7 ms |
| One forced rejection at draft position 2 | Exact | 8/10 | 1875.4 ms |

The speculative totals include prefill, the proposer callback, verification,
host acceptance, and fallback. The perfect run spent 1163.5 ms in prefill,
297.4 ms in verifier graphs, and 0.001 ms in the callback. The rejected run
spent 1199.9 ms in prefill, 673.9 ms in verifier graphs, and 0.004 ms in the
callback. The proposer reads the already computed greedy IDs, so these are
optimistic illustrative timings, not a bound on real draft-source latency.
They exclude the cost of producing those oracle IDs. A copy or CTC proposer
needs a separate full-cost benchmark before runtime enablement. One run does
not establish stable throughput or energy savings. Vulkan and Pixel
measurements remain open.
