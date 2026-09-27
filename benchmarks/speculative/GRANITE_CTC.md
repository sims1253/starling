# Native Granite CTC draft extraction (#313)

The optional BPE CTC head from the pinned [Granite Speech 4.1 2B
revision](https://huggingface.co/ibm-granite/granite-speech-4.1-2b/tree/de575db64086f84fdc79da4932d1076e965bc546)
can now be exported into the Granite GGUF. The normal model loads without
that head. `extract_ctc_draft` runs the native encoder once, returns the
pre-feedback middle-layer grapheme logits and final hidden state as one graph
output, pools four frames using `1 - blank_probability`, applies `out_llm`,
then collapses repeat and blank labels to tokenizer IDs. This follows
[`CTCBPEDraft`](../../src/starling/granite/speculative.py), including the
BF16 rounding of pooled activations and head logits, then FP32 softmax before
argmax. The model-specific C probe exposes IDs to the parity runner; ordinary
greedy decode is unchanged.

Build with the repository's native CMake configuration, then convert the
pinned source snapshot with the opt-in flag:

```sh
uv run --with gguf python scripts/convert_granite_gguf.py \
  --snapshot /path/to/de575db64086f84fdc79da4932d1076e965bc546 \
  --include-ctc-head --output /tmp/granite-with-ctc.gguf
cmake -S backends/native -B /tmp/granite-build -DSTARLING_GGML_SHARED=ON
cmake --build /tmp/granite-build --target starling_ggml
STARLING_GGML_DEVICE=cpu uv run python benchmarks/speculative/eval_granite_ctc.py \
  --library /tmp/granite-build/libstarling_ggml.so \
  --gguf /tmp/granite-with-ctc.gguf \
  --wav tests/fixtures/2086-149220-0033.wav \
  --reference benchmarks/speculative/granite_ctc_librispeech_reference.json
```

The Python reference IDs were produced on the source BF16 model with
`FusedEncoder(mode="eager")`, `CTCBPEDraft`, and `load_out_llm` in the same
repository, against the two SHA-256-pinned public WAVs in the reference JSON
files. On CPU native GGUF, extraction matched **26/26 IDs** for the 7.435 s
LibriSpeech fixture and **86/86 IDs** for the 24.94 s multilingual sample
from the model repository. The full CTC-inclusive GGUF has 942 tensors and
is 205,723,776 bytes larger than the local pre-existing base GGUF; the
optional head's source file is 205,723,810 bytes. These are exact token-ID
comparisons, not just transcript similarity.
After the FP32 softmax argmax correction, both CPU comparisons still match
exactly (26/26 and 86/86) with native library SHA-256
`4a911dffc5a81e3a230d8eaf0aeb7c7d980372c2371b03db52660a28fd2b89cf`.

Each reference also pins the SHA-256 of the CTC-inclusive GGUF used for the
reported comparison (`cec47a4fb872ace2713447409f01e0f9ef0d2a7f217db8bb9c33ea0ef96c916a`).
The source `out_llm.safetensors` at the pinned revision has SHA-256
`6cc10d68fe05aec359aceffd597617c875b23f27211ee6dcdb7510d9e90fc64e`.
The parity runner rejects a different GGUF before loading it and prints the
actual GGUF hash, library hash, and selected native backend with the ID result.
Thus the result is tied to the supplied binary and artifact, while a newly
converted GGUF needs its own pinned reference if its bytes differ.

The C probe accepts a zero-capacity call to report the required token count,
but that call performs a full encoder and head pass. Callers that know the
audio duration should allocate a conservative token buffer before the first
call to avoid running the encoder twice.

Native argmax now chooses the first label when FP32 softmax probabilities tie,
as the Python reference does. Distinct BF16 logits can become equal after
softmax, including at the smallest BF16 subnormal. The CPU CI regression
covers five small-vocabulary frames and a 100353-label frame.

This is an extraction/parity milestone. The generic #311 batched verifier is
still needed to consume the draft, prove output equality to target-only
greedy, and measure CPU/Vulkan tokens/s. The present probe runs a one-shot
encoder graph and reads back the middle/output bundle; its speed does not
predict an integrated fused draft path. Long audio needs a draft per policy
chunk. Parakeet text did not draft standalone MOSS effectively in the
[12-clip FLEURS study](../fast_engine/RESEARCH_LOG.md) (1.03–1.05 output
tokens/pass); that does not measure a MOSS final conditioned on live Parakeet
partials in staging. No Pixel latency/energy, native acceptance, or runtime
enablement is claimed here.
