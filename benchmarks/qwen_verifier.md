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
The 12-token MOSS timing predates the CPU verifier correction below and is
historical oracle evidence, not a current performance measurement.

## CPU verifier attention-width correction

A later real Granite Speech 4.1 2B check exposed a token-parity failure in
the original CPU verifier. Greedy CPU decoding attended only to its populated
KV prefix, while the batched verifier attended to the full cache width with
future keys masked. On the model-supplied 24.94-second `multilingual_sample.wav`
clip, greedy emitted 98 IDs and the original verifier emitted 100 IDs at
K=1, 2, and 4: it inserted token ID 11 twice. The clip came from model
snapshot `de575db64086f84fdc79da4932d1076e965bc546` and has SHA-256
`91d243650809c1274141ec20ff23045315eaf27567694002ea3ef390048b7058`.
The optional-CTC-head GGUF used in that check has SHA-256
`cec47a4fb872ace2713447409f01e0f9ef0d2a7f217db8bb9c33ea0ef96c916a`.

The CPU verifier first copied the S candidate KV rows and bounded its
attention tensor to `past + S` keys, the extent of the final row. That fixed
the observed Granite case: K=1, 2, and 4 each produced the same 98 IDs and
stop result as greedy on that clip. Earlier rows still reduced over the batch
width with future keys masked, whereas greedy used a separately sized tensor
at each token. The [medium MOSS/LibriSpeech pilot in #347](https://github.com/sims1253/starling/pull/347)
exposed the remaining gap:
greedy stopped after 89 IDs, while a K=2 Parakeet-text copy draft produced
91 IDs, first differing at zero-based index 21.

The CPU verifier now reduces each query row over only `past + row + 1` keys
by default, while keeping batched projections and KV writes. On that public
MOSS pilot, two warmed alternating pairs each at K=2 and K=4 for both short
and medium fixtures matched every greedy ID, EOS, and saved text (eight pairs
total). The measured full MOSS calls were 21.7–26.0% faster with the draft;
the saved final Parakeet transcript was assumed already available, so live
preview accuracy and any new Parakeet inference cost are outside that pilot.
`STARLING_MOSS_VERIFY_BATCH_ATTN=1` restores the earlier bounded-batch
reduction only for diagnosis. `STARLING_GRANITE_FULLCAP=1` retains the
full-capacity verifier when greedy explicitly selects full-capacity
attention; the GPU verifier also keeps its existing graph. The real Granite
check uses the
`granite_ctc_verify_test` harness from the stacked #313 work, which adds a
CTC draft source and is not part of the generic verifier. The self-synthesized
`speculative_verifier_test` also passed after the per-row change. These sampled
cases do not prove greedy parity for every model, audio input, or device;
runtime enablement remains gated on the intended workload and hardware.
