# Native Granite CTC verification

This opt-in path joins the native CTC draft extractor from #334 to the causal
batched verifier from #338. A single encoder graph returns both the projector
embeddings and the middle/final encoder tensors needed by the CTC head. The
head produces tokenizer IDs; `CtcProposer` reads only those IDs and the
target's *committed* prefix. It searches forward up to 40 CTC IDs after a
target correction or bonus token, and adapts the proposal length from 1 to
the configured cap. The verifier still chooses every emitted token. Ordinary
Granite transcription remains greedy by default.

The research C symbol `starling_ggml_granite_decode_ctc(handle, pcm, n,
max_k, err)` accepts a model-specific `starling_ggml_granite_load` handle,
not a common `starling_ggml_ctx`. The caller frees its returned string with
`starling_ggml_free_string` and the handle with
`starling_ggml_granite_free`. It shares the ordinary C entry point's padded
chunk and token-budget loop. A GGUF without the optional CTC head returns a
clear error; it does not change the default decoder. This model-specific C
symbol has no cancellation argument. The raw Granite handle has no whole-call
runtime lock: callers must serialize all raw-handle load, draft, decode, and
free calls across handles, avoid overlapping them with common-API model calls
or global shutdown, and free raw handles before `starling_ggml_shutdown`.
Read error strings before the next call or free. The C++ function accepts `CancelCheck`
and discards output when cancelled before or after draft extraction; the
generic verifier also checks cancellation after each graph.

The first integrated prototype ran `encode_audio_and_project` and
`extract_ctc_draft` separately. It reran the 16-layer encoder for every
candidate decode. The shared graph now returns the CTC bundle and projector
embeddings together as one explicit output. The old separate path remains an
independent parity oracle; the shared path has bitwise-equal projector values
and exact CTC IDs on both public clips below.

The first CPU correction bounded the batched verifier's attention to
`past+S` keys. It restored parity on the 24.94 s sample after full-capacity
masked attention inserted two comma IDs. A later MOSS case exposed the
remaining width difference: earlier batch rows still reduced over `past+S`,
while greedy row `j` reduced over `past+j+1`. The current CPU verifier reduces
each row over its own populated prefix. The GPU graph is unchanged, and the
sampled CPU results below do not prove universal numerical parity.

The real model is the optional-head GGUF pinned by
[`granite_ctc_librispeech_reference.json`](granite_ctc_librispeech_reference.json)
and [`granite_ctc_multilingual_reference.json`](granite_ctc_multilingual_reference.json)
(SHA-256 `cec47a4fb872ace2713447409f01e0f9ef0d2a7f217db8bb9c33ea0ef96c916a`).
The 7.435 s LibriSpeech WAV has SHA-256
`5fceacff0315d49cb59fcc505bcecf1ed5f2f35c2897b1e65a59f30e5d922150`;
the 24.94 s `multilingual_sample.wav` from the model repository has SHA-256
`91d243650809c1274141ec20ff23045315eaf27567694002ea3ef390048b7058`.
The latter is one supplied clip; its name does not establish coverage of
multiple languages.

Build and run with the optional-head GGUF and the two public WAVs:

```sh
cmake -S backends/native -B /tmp/granite-ctc-verify -DSTARLING_GGML_SHARED=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/granite-ctc-verify --target starling_ggml granite_ctc_proposer_test speculative_verifier_test granite_ctc_fused_parity_test granite_ctc_verify_test granite_ctc_chunk_test granite_ctc_entry_test
STARLING_GGML_DEVICE=cpu STARLING_GGML_THREADS=4 taskset -c 8-15 /tmp/granite-ctc-verify/granite_ctc_fused_parity_test /path/to/ctc.gguf tests/fixtures/2086-149220-0033.wav
STARLING_GGML_DEVICE=cpu STARLING_GGML_THREADS=4 taskset -c 8-15 /tmp/granite-ctc-verify/granite_ctc_verify_test /path/to/ctc.gguf tests/fixtures/2086-149220-0033.wav 2
STARLING_GGML_DEVICE=cpu STARLING_GGML_THREADS=4 taskset -c 8-15 python benchmarks/speculative/eval_granite_ctc_verify.py --library /tmp/granite-ctc-verify/libstarling_ggml.so --gguf /path/to/ctc.gguf --wav tests/fixtures/2086-149220-0033.wav --max-k 4 --repeats 2
```

The following original CPU results used the bounded-batch verifier and are
preserved as historical measurements. The verifier test warms greedy and
K=1/2/4 once, then measures two adjacent
pairs at K=2 and K=4. The first pair runs greedy then CTC; the second reverses
the order. `full_ms` includes mel, the shared encoder/projector graph, CTC
pool/head, prompt embeddings, and verification. The `RUN` records also expose
each stage, accepted/proposed counts, verifier calls, fallback steps, exact ID
parity, and stop reason. Model load is outside `full_ms`. First-use runs are
kept separate from the paired comparison.

The 7.435 s LibriSpeech clip produced 26 exact CTC draft IDs and 32 greedy
output IDs. Projector embeddings were bitwise-equal to the ordinary path;
cancelling after draft extraction returned no output. Each measured CTC run
had the same 32 IDs and EOS as its neighboring greedy run:

| K | Order | Greedy full | CTC full | Direct accepts |
|---:|:---|---:|---:|---:|
| 2 | greedy → CTC | 6.444 s | 5.428 s | 14/23 |
| 2 | CTC → greedy | 6.379 s | 5.263 s | 14/23 |
| 4 | greedy → CTC | 6.303 s | 5.442 s | 14/25 |
| 4 | CTC → greedy | 6.470 s | 5.412 s | 14/25 |

The raw ID and stage records are in
[`granite_ctc_verify_short_cpu.log`](granite_ctc_verify_short_cpu.log).
The public C entry points returned the same transcript SHA-256
`41f9e18f92ad8dfa94861d37ccb1262f07a1660c7b42f50bf51252cf5b5d0383`
in both orders: greedy 8.123 s then CTC 6.817 s, and CTC 6.499 s then
greedy 8.211 s. Their complete stage and artifact log is
[`granite_ctc_verify_short_capi_cpu.log`](granite_ctc_verify_short_capi_cpu.log).

The 24.94 s model-repository clip produced 86 exact CTC draft IDs and 98
greedy output IDs. The native fused graph's projector values were bitwise
equal to the ordinary encoder/projector graph. Each measured CTC run below
had the same 98 IDs and EOS as its neighboring greedy run:

| K | Order | Greedy full | CTC full | Direct accepts |
|---:|:---|---:|---:|---:|
| 2 | greedy → CTC | 20.833 s | 17.243 s | 51/73 |
| 2 | CTC → greedy | 20.850 s | 16.729 s | 51/73 |
| 4 | greedy → CTC | 20.760 s | 16.633 s | 58/83 |
| 4 | CTC → greedy | 21.075 s | 16.614 s | 58/83 |

The CPU bounded-width correction was necessary for this historical result.
Before it, the same clip's greedy output was stable at 98 IDs while the
full-capacity CPU verifier produced 100 IDs with two inserted token-11 commas
at K=1, 2, and 4.
After the correction, all three warm K runs matched 98/98 IDs. The input
audio and draft IDs did not change.

For long-audio policy, a 32.375 s WAV formed by concatenating the two pinned
public WAVs (SHA-256
`835681d10d7ad06169e2b5fe176c227faf99ace387ff1066b9edf014c60a67bf`)
split into a 30 s chunk and a 2.375 s tail padded to 30 s. At K=4, chunk 1
matched all 121 greedy IDs; chunk 2 matched all 12. The manual ID test's
historical raw log labels `greedy_full_ms` and `ctc_full_ms` are asymmetric:
the greedy clock includes mel extraction, but the CTC clock starts after
greedy and reuses its mel. The source now labels them `greedy_from_pcm_ms`
and `ctc_from_reused_mel_ms`; those manual clocks establish parity and stage
costs, not a full-cost speed comparison. The raw-WAV C entry points each
recompute mel and cover the whole chunk loop. Their first chunks took
25.518 s greedy and 21.545 s CTC; their padded tails took 15.128 s greedy
and 15.398 s CTC. On this single run the tail was 0.270 s slower with CTC:
its full padded encoder/head cost bought few draftable output tokens. The
complete request took 40.648 s greedy and 36.944 s CTC, with identical
UTF-8 transcript SHA-256
`71bea620ec405ca1b0bf8d0a1e36000ebc6bb8a7bebeaf255ef388dfea374089`.
That C request has one run per mode, so it is a parity check and provisional
timing, not a repeated long-audio speed estimate.

The raw stage and parity records are in
[`granite_ctc_verify_model_sample_cpu.log`](granite_ctc_verify_model_sample_cpu.log),
[`granite_ctc_verify_long_ids_cpu.log`](granite_ctc_verify_long_ids_cpu.log),
and [`granite_ctc_verify_long_capi_cpu.log`](granite_ctc_verify_long_capi_cpu.log).
The C benchmark prints the actual GGUF, library, and WAV SHA-256 hashes and
backend. `STARLING_GRANITE_TIMING=1` prints one `GRANITE_CTC` stage line per
chunk; its `ctc_head` field includes pooling and the optional BPE head.
The measured historical C shared library's SHA-256 was
`c9f65e6cfa702f36fc62ebcd61bc42070fd9037cfdcb0bec16e1b81b54ee5687`.
The short ID-level benchmark binary was
`dc59043a28cb8560910d3bc510f727adc20e363e2af20e2b6e2ee4b34b1a82c3`;
the longer model-sample ID benchmark binary (built before the opt-in C entry
point was added, with the same shared encoder and CPU verifier code) was
`33d611385dd1a120315d27adb48befe9430eb872a81acc8761f2703f2cc97a40`.

## CPU per-row verifier revalidation

The current CPU default reduces each verifier row over `past+row+1` keys,
matching greedy's populated attention width for that row. Batched projection
and cache operations still differ from greedy, so exact IDs on these fixtures
are measured evidence, not a guarantee for other audio. This source is commit
`5838bad` in this stack (the same default as #338 commit `419723d`); the
measured shared library is SHA-256
`bfa33b547b582afc1fa858416c9ef8f2e6117cb34a9d8a8472c8613f0378530b`
and the ID benchmark binary is
`e852b005a22ea1e72f7760cb3661e7667fee364b977b6de27aaca8c207faedeb`.
The GGUF and WAV hashes remain those above. All runs selected backend CPU,
used four ggml threads, and were pinned to logical CPUs 8–15. The test warmed
greedy and K=1/2/4, then alternated execution order for two K=2 and two K=4
pairs. Every CTC run matched greedy IDs and EOS: 32/32 on LibriSpeech and
98/98 on the model-repository clip; draft IDs remained 26 and 86. Each
`full_ms` below starts before mel and includes CTC extraction, proposal, and
verification when applicable. Model load is outside the clock.

| Clip | K | Order | Greedy full | CTC full | Direct accepts |
|:---|---:|:---|---:|---:|---:|
| 7.435 s | 2 | greedy → CTC | 6.548 s | 5.486 s | 14/23 |
| 7.435 s | 2 | CTC → greedy | 6.283 s | 5.303 s | 14/23 |
| 7.435 s | 4 | greedy → CTC | 6.099 s | 5.381 s | 14/25 |
| 7.435 s | 4 | CTC → greedy | 6.353 s | 5.580 s | 14/25 |
| 24.94 s | 2 | greedy → CTC | 20.492 s | 17.646 s | 51/73 |
| 24.94 s | 2 | CTC → greedy | 20.190 s | 16.928 s | 51/73 |
| 24.94 s | 4 | greedy → CTC | 20.412 s | 16.740 s | 58/83 |
| 24.94 s | 4 | CTC → greedy | 20.774 s | 16.721 s | 58/83 |

The complete new ID/stage records are
[`granite_ctc_verify_short_cpu_perrow.log`](granite_ctc_verify_short_cpu_perrow.log)
and [`granite_ctc_verify_model_sample_cpu_perrow.log`](granite_ctc_verify_model_sample_cpu_perrow.log).
The same 32.375 s concatenated WAV again produced exact per-chunk target IDs
at K=4: 121/121 on the 30 s first chunk and 12/12 on the padded tail. Its
manual [ID log](granite_ctc_verify_long_ids_cpu_perrow.log) labels the CTC
clock `ctc_from_reused_mel_ms`, so it is not a full-cost comparator.
Through the raw-WAV C API, both modes returned the same transcript SHA-256
`71bea620ec405ca1b0bf8d0a1e36000ebc6bb8a7bebeaf255ef388dfea374089`.
This single whole-request run took 40.497 s greedy and 37.201 s CTC. The
first chunk took 25.643 s versus 21.894 s; the padded tail took 14.854 s
versus 15.306 s, respectively. These C timings include mel on both sides;
the padded-tail difference is one observation, not a latency distribution.
The [raw C API log](granite_ctc_verify_long_capi_cpu_perrow.log) pins the
GGUF, library, and WAV hashes and reports the per-chunk stages.

The two asset-free CTC proposer/entry tests run in native CI; tests requiring
the optional-head GGUF and public WAVs remain manual. The CPU pilot is not a
runtime enablement gate. It uses one LibriSpeech clip
and one supplied model-repository clip, with four ggml threads pinned to four
physical cores (logical CPUs 8–15). Vulkan, Pixel stop-to-final latency,
energy, and the #310 dictation set remain unmeasured. Parakeet-to-MOSS drafts
are a separate proposer and cost model; no Parakeet run is counted as free in
this Granite result.
