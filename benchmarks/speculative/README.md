# Copy draft acceptance probe (#312)

`copy_draft.py` proposes exact target-token IDs from the raw transcript. It
tracks a monotone position in the source, skips tokens omitted by the target,
and uses an n-gram from already emitted output when the position stalls. The simulator
shows only emitted target IDs to the proposer. It checks proposals against a
recorded greedy ID stream and counts verify passes, without running a model.

Run the S1-mini study after capturing stock goldens with
`python -m starling.s1.golden`:

```sh
python benchmarks/speculative/eval_copy_drafts.py --golden-dir golden/s1
```

The checked local stock goldens were produced by the repository's
[`src/starling/s1/golden.py`](../../src/starling/s1/golden.py) procedure from
the short, medium, and long transcripts in
[`tests/fixtures/s1_transcripts.py`](../../tests/fixtures/s1_transcripts.py).
The files are gitignored; the JSON output prints each file's SHA-256 so a
later run can establish whether it used the same target IDs. The source was
tokenized with the cached `superwhisper/s1-mini` tokenizer. This table records
the 2026-09-28 run at max K=2:

| Transcript | Source tokens | Greedy output tokens | Verify passes | Accepted draft tokens/pass | Output tokens/pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 19 | 11 | 8 | 0.375 | 1.375 |
| medium | 77 | 79 | 53 | 0.491 | 1.491 |
| long | 257 | 281 | 188 | 0.495 | 1.495 |

The target stream is reproduced by the oracle replay, which is a simulator
property; this replay alone establishes neither native numerical parity nor a
speedup. The [native verifier in #338](https://github.com/sims1253/starling/pull/338)
and [S1 copy pilot in #345](https://github.com/sims1253/starling/pull/345)
now provide that next, separate step. With the per-row CPU verifier, the
native pilot matched greedy and stock IDs/text on all three goldens and
preserved both spans of a synthetic protected fixture. Two warmed paired CPU
repeats per case favored copy at K=2 on the stock goldens and K=4 on the
protected fixture; the measured copy/greedy ratios were 0.66–0.92×. The
earlier full-capacity CPU verifier had both a real Granite token-parity
counterexample and slower S1 copy timings. A later bounded-batch verifier was
faster on S1 but failed a medium MOSS parity case. #345 keeps both historical
runs separate from the current per-row pilot.

Runtime default enablement still needs the representative #310 dictation
workload, Pixel and fast-engine parity/cost, and target-device latency and
energy gates. This offline probe can also take a previous processed output as
its `source` for a revision study once paired revised-output goldens exist.
