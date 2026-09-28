# Native S1 aligned-copy pilot (#312)

`STARLING_S1_COPY_DRAFT=1` opts S1 normalization into the real batched Qwen
verifier from [#338](https://github.com/sims1253/starling/pull/338). The
S1's C API tokenizes the raw transcript with its BPE and passes those IDs to
[`CopyDrafter`](../../cpp/s1/copy_draft.hpp). The proposer reads only that
source and the verified output prefix. It follows a monotone source position,
skips omitted filler tokens within a 32-token window, and tries a recent
output n-gram if alignment stalls. It starts at K=2 (or K=1 when capped there),
grows K after a fully accepted proposal, and halves K after a rejection.
`STARLING_S1_COPY_MAX_K` is 1–16; the pilot uses 1, 2, and 4. The default
normalization path remains greedy. The optional `STARLING_S1_DUMP_IDS` file
contains only verified target IDs and is intended for parity testing.

The source-only proposal algorithm follows the offline
[#327 study](https://github.com/sims1253/starling/pull/327). The native pilot
executes the full S1 pipeline, including BPE encoding, embedding lookup,
target prefill and decoding, and detokenization. Each candidate uses the same
source transcript and model. Each timed call also writes one diagnostic ID
file, in both arms. The benchmark loads the GGUF once, warms each greedy and K
configuration, then times alternating paired C API calls. It checks
byte-exact generated IDs against native greedy for three captured stock S1
goldens and one public protected-span fixture. The first three also compare
native IDs and text with the locally captured stock Transformers goldens.

```sh
cmake -S backends/native -B /tmp/starling-312-native-build \
  -DSTARLING_GGML_SHARED=ON -DSTARLING_GGML_CUDA=OFF
taskset -c 0-7 cmake --build /tmp/starling-312-native-build \
  --target starling_ggml s1_copy_draft_test -j 4
/tmp/starling-312-native-build/s1_copy_draft_test
STARLING_GGML_THREADS=4 taskset -c 0-7 python3 benchmarks/speculative/eval_native_s1_copy.py \
  --lib /tmp/starling-312-native-build/libstarling_ggml.so \
  --gguf /path/to/s1-bf16.gguf --golden-dir /path/to/golden/s1 \
  --cases short medium long --ks 2 --repeats 2
STARLING_GGML_THREADS=4 taskset -c 0-7 python3 benchmarks/speculative/eval_native_s1_copy.py \
  --lib /tmp/starling-312-native-build/libstarling_ggml.so \
  --gguf /path/to/s1-bf16.gguf --golden-dir /path/to/golden/s1 \
  --cases protected --ks 4 --repeats 2
```

The protected fixture contains a synthetic code and public example URL; it
is a parity check, not a claim that the model preserves those spans. The
captured goldens are gitignored and the benchmark reports their SHA-256
hashes. No personal transcript or audio is used here.

## Results

### Historical runs before the #338 CPU width correction

Before #338 matched the CPU verifier's attention width to greedy, an
exploratory first-use CPU pass ran each case's greedy baseline before K=1,
2, and 4 without a warmup for each copy configuration. It is evidence of
token parity, not a stable speed estimate. That early binary also wrote the
greedy diagnostic ID file twice per call; the paired runs below use one write
per arm. It used the BF16 S1 GGUF with
SHA-256 `c4fc61df01df19655d796e085d29aa41647d33cfd090824778ecadd76a2bb11c`.
The host is an AMD Ryzen 9 5900X. The exploratory pass used CPUs 0–15;
the later paired pass used CPUs 0–7 and four ggml threads. The protected
fixture's UTF-8 source SHA-256 is
`23ebf0fce659f72663040be0939485f46bd8af612fd1c4535f2012cbb8d1a8b6`.
The native build uses Starling's pinned ggml commit `e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`
with its required local patch series.
All 12 copy runs matched native greedy IDs and text. The short, medium, and
long greedy and copy IDs also matched their stock goldens; the synthetic
protected code and URL survived in both outputs. The captured stock artifacts
are:

| Stock tier | Generated IDs | Golden ID file SHA-256 |
| --- | ---: | --- |
| Short | 11 | `f77e234befdbe42a8c632d104cbffdee994b09f54d9ec3c7b38a4b85af3830cb` |
| Medium | 79 | `b65f89c10764635030c93b30bf263079e68889a464a79b817a5ab6c3409e5aac` |
| Long | 281 | `22d880e3ec82b152e1d7fac7a89a87feca64530b3a32933f0ab7f669fefafc7e` |

In that pass, the call times were:

| Case | Greedy | K=1 | K=2 | K=4 |
| --- | ---: | ---: | ---: | ---: |
| Short | 0.78 s | 1.17 s | 1.19 s | 1.21 s |
| Medium | 4.87 s | 8.47 s | 7.52 s | 7.36 s |
| Long | 18.68 s | 26.79 s | 25.22 s | 25.43 s |
| Protected | 1.56 s | 2.00 s | 1.59 s | 1.29 s |

The historical paired runs used native library SHA-256
`fa4098f4fb7ff33ef70591c569c7f67680ac0fcff8fe8824d77dcd87844ab521`
on the CPU backend. The harness performs one untimed warmup of each greedy and
K configuration, then alternates their order in two paired repeats. Its
[stock results](results/native_s1_copy_cpu_stock_pre_width_fix.json) and
[protected results](results/native_s1_copy_cpu_protected_pre_width_fix.json) record the model
and library hashes, actual backend name, source and golden hashes, times,
verified token ID hashes, text parity, and protected-span results for each
pair. The timed C API call includes one diagnostic ID file write in each arm.

| Case | K | Greedy repeat 1 / 2 | Copy repeat 1 / 2 | Copy / greedy repeat 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| Short | 2 | 1.16 / 1.07 s | 1.94 / 1.90 s | 1.68 / 1.78× |
| Medium | 2 | 5.40 / 6.32 s | 10.25 / 10.34 s | 1.90 / 1.64× |
| Long | 2 | 22.58 / 28.61 s | 38.51 / 40.78 s | 1.71 / 1.43× |
| Protected | 4 | 2.00 / 1.85 s | 2.36 / 2.57 s | 1.18 / 1.39× |

All eight paired comparisons matched native greedy IDs and text. Every stock
call also matched its captured stock IDs and text. Both protected spans were
present in greedy and copy output on both repeats. The copy callback took
under 0.4 ms per stock call; low acceptance (3/10 short, 26/70 medium,
93/246 long) and full-capacity verifier work dominated. All four cases were
slower with copy drafting under that historical verifier. They do not describe
the corrected CPU verifier's cost.

An intermediate paired run used the same harness but overlapped a Granite
pilot on CPUs 0–15 and began with the ggml physical-core default thread count.
It was contention-affected and is excluded from the table; the historical
paired runs used exclusive CPUs 0–7 and
`STARLING_GGML_THREADS=4`.

### Historical bounded-batch CPU run

The S1 branch was then rebased onto #338 head
`b967ef23c734c5e16f4a52865af96e61e86c8214` and its native library
rebuilt. That library SHA-256 is
`c74f9f1d74f6b5567bdb02c3ad5568389a16f04a5f1e9484e96e499f44014d0c`.
The model, transcripts, golden files, harness, CPU set 0–7, four ggml
threads, untimed warmups, alternating call order, and two repeats per case
match the earlier paired method. The [bounded-batch stock
results](results/native_s1_copy_cpu_stock_batch_width.json) and [bounded-batch
protected results](results/native_s1_copy_cpu_protected_batch_width.json)
contain each trial's
timings, library/model/source hashes, backend name, and parity checks. The
timed call includes BPE tokenization, embedding lookup, the complete target
generation, detokenization, and one diagnostic ID write on each arm; model
load is reported separately.

| Case | K | Greedy repeat 1 / 2 | Copy repeat 1 / 2 | Copy / greedy repeat 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| Short | 2 | 1.03 / 0.92 s | 0.87 / 0.87 s | 0.85 / 0.94× |
| Medium | 2 | 4.70 / 6.42 s | 3.86 / 3.92 s | 0.82 / 0.61× |
| Long | 2 | 19.37 / 27.81 s | 15.22 / 16.24 s | 0.79 / 0.58× |
| Protected | 4 | 1.80 / 1.86 s | 1.15 / 1.14 s | 0.64 / 0.61× |

All eight bounded-batch verifier pairs matched native greedy IDs and text. The
short, medium, and long pairs also matched their captured stock IDs and text.
The synthetic code `ZX-1042` and URL `https://example.org/manual` survived in
both greedy and copy outputs on both protected repeats. The copy callback
itself took at most 0.4 ms per call; measured acceptance was 3/10 short,
26/70 medium, 93/246 long, and 17/19 protected. That bounded-batch CPU
verifier made copy faster in each of these pairs, including the proposer's
full cost. It was later replaced by per-row attention after the [MOSS pilot
#347](https://github.com/sims1253/starling/pull/347) found a parity
counterexample. These results do not describe the current verifier.

### Current per-row CPU verifier pilot

The S1 branch was rebased onto the #338 per-row CPU verifier runtime commit
`419723df29f2990f2a22efb58e256d8b848bee24`. Its rebuilt native library
SHA-256 is
`77eb84067a8de637f828633c9fb3f50bd1763afca81394f1545133f07d7e39fc`.
The same BF16 GGUF, transcript and golden hashes, four ggml threads, CPU set
0–7, one warmup per arm, and alternating paired repeats were used. The
[current stock results](results/native_s1_copy_cpu_stock.json) and [current
protected results](results/native_s1_copy_cpu_protected.json) include every
trial's timings, backend, artifact hashes and parity flags. The timed C API
call includes source tokenization, embedding lookup, complete target decode,
detokenization and a symmetric diagnostic ID write; model load is separate.

| Case | K | Greedy repeat 1 / 2 | Copy repeat 1 / 2 | Copy / greedy repeat 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| Short | 2 | 0.94 / 0.95 s | 0.83 / 0.84 s | 0.88 / 0.89× |
| Medium | 2 | 4.26 / 4.45 s | 3.50 / 3.51 s | 0.82 / 0.79× |
| Long | 2 | 18.01 / 17.51 s | 15.58 / 16.07 s | 0.87 / 0.92× |
| Protected | 4 | 1.54 / 1.54 s | 1.02 / 1.04 s | 0.66 / 0.67× |

All eight pairs matched native greedy IDs and text. The three stock tiers
also matched their captured Transformers IDs and text. Both the code and URL
survived in greedy and copy output on both protected repeats. Copy was faster
in each sampled CPU pair, but these two repeats per case do not establish a
stable latency distribution or general parity for other inputs. The feature
remains opt-in.

The measured CPU pilot result does not establish Pixel
latency, energy, or p50/p95 on the [#310](https://github.com/sims1253/starling/issues/310)
workload. S1 fast-engine integration, an instruction-model rewrite/translate
trial, and a previous-output revision source remain separate work. The
`CopyDrafter` constructor accepts arbitrary source token IDs, so a later
revision caller can supply a previous processed output without changing the
proposal algorithm. No default-on policy is proposed without a latency and
energy win on the target device.
