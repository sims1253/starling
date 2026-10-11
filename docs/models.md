# Models

Starling supports the following speech recognition models. Each has an
in-tree ggml engine in `libstarling_ggml` that `starling-serve` loads by slug
(see [native serving](native-serving.md)); GGUFs come from the converters in
`scripts/convert_*_gguf.py`. S1-mini, listed last, edits transcripts after
recognition.

- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (slug `parakeet`): FastConformer + TDT transducer (no LLM). Also has a hand-written Vulkan fast engine ([fast engines](fast-engine.md)).
- [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) (slug `granite`): CTC conformer encoder + BLIP2 Q-Former projector + granite-4.0-1b decoder, greedy path.
- [`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B) (slug `moss`): Qwen3-omni MoE encoder + Qwen3 decoder. Also has a Vulkan fast engine.
- [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (slug `qwen3`): Whisper-style windowed-attention encoder + Qwen3 decoder, greedy path.
- [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) (slug `qwen3_06`): smaller sibling of the 1.7B (narrower 18-layer/896-wide audio tower, 1024-wide decoder trunk under the same 16Q/8KV×128 attention). Byte-identical tokenizer/chat template/processor to the 1.7B, so it rides the same engine with every dim derived from the checkpoint.
- [`AutoArk-AI/ARK-ASR-3B`](https://huggingface.co/AutoArk-AI/ARK-ASR-3B) (slug `ark`): Whisper encoder + MLP adapter + Qwen2.5 decoder.
- [`Audio8/ARK-ASR-0.6B`](https://huggingface.co/Audio8/ARK-ASR-0.6B) (slug `ark06`): distilled 0.6B sibling on the ark engine (Qwen2.5-0.5B-class decoder), 19 languages, Apache-2.0. Parity pending GPU verification.
- [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602) (slug `voxtral`): the first streaming-native model in the repo: a causal conv embedder + 32-layer sliding-window (750) encoder + ×4-merge GELU projector into a Ministral-3-class decoder (26 layers, GQA, sliding window 8192), with additive audio injection at every position and an AdaRMSNorm delay conditioning (default 6 tokens ≈ 480 ms) on the MLP branch. Decode runs in lockstep with audio positions (12.5 tok/s length cap). 13 languages, Apache-2.0. Parity pending GPU verification.

The autoregressive models (granite, moss, qwen3, qwen3_06, ark, ark06) share an
encoder + LLM-decoder pattern where the decode loop is the bottleneck. Parakeet
is a transducer; voxtral decodes text in lockstep with streaming audio
positions.

- [`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini) (slug `s1`): text-to-text: a 0.6B Qwen3 decoder-only normalizer that rewrites raw ASR transcripts as clean written text (removes fillers, resolves self-corrections, punctuation/truecasing, numbers/emails under a `[Styling|Structure|Context]` control line). No audio front-end: the engine has a C++ BPE encoder and a text path via `starling-serve`'s `POST /normalize`, byte-exact against stock transformers on all fixture tiers. Apache 2.0 with a naming clause ("S1-mini by Superwhisper").

Not yet native:

- [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b): Unified FastConformer-RNN-T. Native support is tracked in issue #358. A NeMo-free PyTorch reference (the `.nemo` checkpoint loaded directly, encoder/prediction-net/joint hand-built, eager greedy RNN-T decode) lives in `scripts/goldens/parakeet_unified_ref/` and captures the goldens a native port must match (`scripts/goldens/parakeet_unified_golden.py`).

Golden references and their capture scripts are listed in
[engine development](ggml-engine.md#goldens).
