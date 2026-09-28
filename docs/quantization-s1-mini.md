# Experimental S1-mini decoder recipe

The `s1-mini-q4-k-m` catalog profile stores decoder linear weights as Q4_K,
with Q6_K for attention value/output and FFN down projections. Its tied
embedding/language-model head stays BF16. The native S1 loader accepts this
opt-in `quantized` numeric profile and rejects unsupported tensor types.
The BF16-exact profile and its default path remain unchanged.

This pilot used `superwhisper/s1-mini` revision
`88f6b15896c73bbb13a3b596e0afe8ea0d5150b4`. The converted BF16 GGUF is
1,198,656,192 bytes (SHA256
`c4fc61df01df19655d796e085d29aa41647d33cfd090824778ecadd76a2bb11c`).
The Q4_K/Q6_K candidate is 610,995,040 bytes (SHA256
`4b12084f0cbbfcfe7c88ed45e8aac28fa20e813faf77f24b9c8d46bc7f9f5834`),
or 587,661,152 fewer stored bytes. The CPU native library SHA256 was
`e078e449497a8b7fa105daeb6b2b72b8fd5ad7d3132775424f17b892f2ad1333`.

Calibration used a Q8_0 intermediate so the CPU imatrix hook could observe
F32 activations at the decoder matmuls. `benchmarks/s1/collect_imatrix.py`
ran the existing five quality prompts, sixteen control combinations, and
three length-tier prompts. The resulting imatrix covered 196 tensors and
203,644 matmul observations (SHA256
`3cfa01fe5b4e82690e8ecb91acb8e2304aa8d41778b89ff1c8a1fc8214f73376`).
The eight protected-span cases in `tests/fixtures/s1_quant_spans.json` were
held out from calibration. On the original host CPU run, all eight candidate
outputs matched the BF16 outputs byte for byte: no pre-existing or new
protected-span violations.
The fixture file SHA256 was
`ef9184b34e51f45ce179026afa675f38975eadae560af1e3a20c00f82cfdb4a5`.

### Pixel 10 Pro CPU pilot (2026-09-28)

The same BF16 and Q4 GGUFs were then loaded in separate native
`starling-serve` processes on a Pixel 10 Pro (Android 17 build
`CP3A.260905.009`). The Android server binary SHA256 was
`e9d65265def7129b0505cea08a1b7749d17e7b1e79645bf880ae84c289bbb224`
from commit `2a3bda4`. Both arms forced the GGML CPU engine with six threads,
used the same eight held-out cases through `/normalize`, and repeated each
case three times. The [phone comparison](evidence/pixel-s1-2026-09-28/README.md)
rechecked every protected span against the fixture definitions.

| Phone result after server warmup | BF16 | Q4_K/Q6_K |
|---|---:|---:|
| Stored GGUF bytes | 1,198,656,192 | 610,995,040 |
| Process smaps PSS before cases | 1,687,097 kB | 1,114,109 kB |
| Process smaps PSS after cases | 1,687,517 kB | 1,114,553 kB |
| New protected-span violations | — | 0 of 8 cases |
| Outputs changed against BF16 | — | 2 of 8 cases |

The warmed Q4 process used 572,988 kB (about 559.6 MiB, 34.0%) less PSS
before the cases. Neither CPU process had a GL mtrack allocation. These are
process-residency snapshots, not loading peaks or a co-residency test. Two
outputs differed: punctuation moved in `negation_and_name`, and “June 2nd”
became “June 2” in `negated_deadline`. All protected spans remained present;
the phone result does not show byte-for-byte output parity. It is a small
English functional pilot, not the broader #310 release gate.

The raw three-repeat per-case HTTP medians ranged from 13.5–35.3 seconds for
BF16 and 2.3–13.6 seconds for Q4. The runs were sequential while the phone
was charging at thermal status 1, with unrelated user activity. These are
observed samples, not a controlled latency or energy comparison. The server's
internal “model loaded” timer (1.7 seconds BF16, 1.3 seconds Q4) ends before
warmup and Q4 weight repacking, so it is not an end-to-end ready-to-serve
load-cost comparison. No co-residency budget has been declared for #229/#295.

These eight English cases are a functional pilot, not a release quality gate.
Issue #310's broader protected-span workload, controlled Pixel latency/energy, and the
co-residency budget from #229/#295 have not been measured. S1-mini remains
English-only. An instruction model is not in this catalog: #295 has a
candidate architecture but has not selected or validated a specific
checkpoint, tokenizer/template, and model for multilingual instructions.

Reproduce with the same model snapshot and a CPU native build:

```bash
python scripts/convert_s1_gguf.py --snapshot /path/to/s1-mini/snapshot \
  --output models/s1-mini-bf16-exact.gguf
build-cpu/starling-quantize --input models/s1-mini-bf16-exact.gguf \
  --output models/s1-mini-q8-calibration.gguf --quant q8_0
python benchmarks/s1/collect_imatrix.py \
  --model models/s1-mini-q8-calibration.gguf \
  --output models/s1-mini-calibration.imx \
  --library build-cpu/libstarling_ggml.so
python -m quants.starling_quants plan s1-mini-q4-k-m \
  --binary build-cpu/starling-quantize \
  --input models/s1-mini-bf16-exact.gguf \
  --output models/s1-mini-q4-k-m.gguf \
  --imatrix models/s1-mini-calibration.imx
python -m quants.starling_quants build s1-mini-q4-k-m \
  --binary build-cpu/starling-quantize \
  --input models/s1-mini-bf16-exact.gguf \
  --output models/s1-mini-q4-k-m.gguf \
  --imatrix models/s1-mini-calibration.imx
```

Run `benchmarks/s1/quant_spans.py run` once per model with `--source` set to
the BF16 GGUF, `--library` set to the same native library, and distinct
`--json` paths; then run `quant_spans.py compare --baseline ... --candidate
...`. The records include exact model, source, engine, and fixture hashes,
backend identity, outputs, and protected-span results. The runner checks that
ctypes loaded the requested native library before inference and hashes that
loaded file. The comparator checks the supplied fixture file and recomputes
span results from each output.
