# #313: already-known Parakeet text as a MOSS draft, desktop pilot

This is a **conditional desktop go for further evaluation**, not a default
feature switch. Native MOSS can use tokenized Parakeet text as a cheap copy
draft when the application already has that text in memory. In a two-fixture
CPU pilot, a corrected per-row verifier matched native greedy MOSS IDs and EOS
in all eight scored pairs, with 22–26% lower *MOSS* full-call time. The source
used here was each fixture's saved **final** Parakeet transcript, an optimistic
proxy for the latest streaming partial. Live-preview completeness, fresh
Parakeet inference cost, broader quality and Pixel latency/energy were not
measured. The benchmark does not wire drafting into the public MOSS API.

## What ran

The source files and hashes, model SHA-256, initial decision rule, amendment,
binary hashes, CPU pinning and exact commands are in
[`moss_preview_copy_prereg.md`](moss_preview_copy_prereg.md). Both WAVs are the
repository's public LibriSpeech fixtures: short is `2086-149220-0033`; medium
tiles that utterance. The source text is the existing Parakeet golden for each
file, normalized to lowercase/collapsed whitespace and encoded with the MOSS
BPE. Its punctuation stays intact; source/target differences are resolved by
monotone alignment against **verified MOSS output IDs** and normal native
verifier rejection. Proposal length adapts up to K=2 or K=4. The drafter sees
no target logits, oracle target IDs or future audio. Both arms redo mel,
encoder/adapter, prompt and full generation to EOS. The draft arm includes
source normalization, BPE encoding and proposer construction in its time.
Model load and file reads are excluded from both arms.

The **first** preregistered run used the then-current CPU batched verifier and
failed at medium K=2. Its greedy output had 89 IDs, matched saved MOSS text
and ended on EOS; the draft had 91 IDs and also ended on EOS. The first ID
divergence was index 21, greedy 13 versus draft 432. K=1 reproduced it. An
empty-proposer control matched all 89 greedy IDs, localizing this observed
failure to batched verification rather than audio prefill or the copy source.
The default verifier had used one `past+S` attention reduction width for all
batch rows. An opt-in experiment gave row `j` its own `past+j+1` key prefix
while keeping batched projections/cache writes. The K=1 control then matched
all 89 IDs. This supports the width explanation for this example; it does not
prove all model/input cases equivalent. The failed records are retained in
[`moss_preview_copy_initial_failed.jsonl`](moss_preview_copy_initial_failed.jsonl)
and the adjacent control JSONL files.

The amended, scored trial enabled that per-row attention path, kept every
other condition and decision rule, and ran two warmed pairs per K and tier.
Pair order reversed on the second repeat. Raw results are in
[`moss_preview_copy_row_attention.jsonl`](moss_preview_copy_row_attention.jsonl).
The intermediate opt-in build is preserved by its executable SHA-256 in the
preregistered amendment; its source commit is no longer reachable. The
[subsequent #338 CPU default](https://github.com/sims1253/starling/pull/338)
selects the same per-row graph without that flag. This report makes no
fresh-run timing claim for the later default build.

| Public fixture | K | Greedy full call, mean | Draft full call, mean | Savings in each pair | Accepted / proposed |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 2 | 6506 ms | 4912 ms | 23.02%, 25.98% | 20 / 20 |
| short | 4 | 6327 ms | 4891 ms | 21.70%, 23.70% | 22 / 23 |
| medium | 2 | 19522 ms | 14947 ms | 23.51%, 23.36% | 57 / 60 |
| medium | 4 | 19588 ms | 14708 ms | 25.13%, 24.70% | 65 / 74 |

Every scored pair matched exact output IDs, decoded text and EOS; greedy also
matched the saved MOSS text for both files. Both K settings exceeded the
preregistered 5% full-call threshold in both repeats and both fixtures. These
are two related clips from one speaker, so the result supports a larger
evaluation, not a general latency or quality claim. Source BPE cost was about
0.18 ms (short) or 0.27–0.30 ms (medium) inside the draft calls. It is valid
to call the *source* free only if a live preview had already run for a reason
other than this draft. For file batch transcription, Parakeet would need to run
and its latency and energy must be added; that path was not measured here.

## Reproduce and next gate

Build the explicit research targets after applying the repository's pinned
`third_party/ggml-patches`:

```sh
cmake -S backends/native -B build-313 -DSTARLING_GGML_TESTS=ON \
  -DGGML_CUDA=OFF -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-313 --target moss-preview-copy-test moss-preview-copy-pilot -j4
./build-313/moss-preview-copy-test
STARLING_GGML_DEVICE=cpu STARLING_GGML_THREADS=4 \
  taskset -c 16-23 \
  ./build-313/moss-preview-copy-pilot MODEL.gguf GOLDEN_DIR \
  tests/fixtures/short.wav tests/fixtures/medium.wav 2
```

On the final #338 CPU code, per-row verifier attention is the default;
`STARLING_MOSS_VERIFY_BATCH_ATTN=1` is a diagnostic opt-out and must stay
unset for this reproduction. The historical scored executable is identified
by the SHA-256 and `STARLING_MOSS_VERIFY_ROW_ATTN=1` flag in the preregistered
amendment. Its intermediate source revision is unavailable, so this branch
can reproduce the final per-row behavior but cannot rebuild that exact binary.

The code path remains research-only. The next evaluation needs real latest
streaming partials, more varied speakers/languages and FLEURS quality, then
paired Pixel 10 Pro final-result latency and energy against the greedy default
under the issue's acceptance gates. There is no phone measurement in this PR.
