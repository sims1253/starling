# Models


Starling supports the following speech recognition models. S1-mini, listed
last, edits transcripts after recognition. See [Python serving](python-serving.md)
and [native serving](native-serving.md) for the models each server accepts.

- [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b): CTC conformer encoder + BLIP2 Q-Former projector + granite-4.0-1b decoder. The in-tree ggml engine (`starling-ggml-granite`, greedy path) runs natively via starling-serve; the Python path adds the optional self-speculative decoding drafting from the encoder's CTC head.
- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): FastConformer + TDT transducer (no LLM). GPU-side mel + chunking for hour-long audio.
- [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b): Unified FastConformer-RNN-T. NeMo-free port: the `.nemo` checkpoint is loaded directly (no `nemo_toolkit`), the encoder/prediction-net/joint are hand-built in PyTorch, and the encoder + greedy RNN-T decode are captured into CUDA graphs.
- [`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B): Qwen3-omni MoE encoder + Qwen3 decoder.
- [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B): Whisper-style windowed-attention encoder + Qwen3 decoder. The in-tree ggml engine (`starling-ggml-qwen3`, greedy path) runs natively via starling-serve.
- [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf): smaller sibling of the 1.7B (narrower 18-layer/896-wide audio tower, 1024-wide decoder trunk under the same 16Q/8KV×128 attention). Byte-identical tokenizer/chat template/processor to the 1.7B, so it rides the same in-tree ggml engine (registry slug `qwen3_06`) and the same CUDA megakernel track (`starling.qwen3_06`) with every dim derived from the checkpoint; CUDA-exact against its golden on short/medium/long.
- [`AutoArk-AI/ARK-ASR-3B`](https://huggingface.co/AutoArk-AI/ARK-ASR-3B): Whisper encoder + MLP adapter + Qwen2.5 decoder.
- [`Audio8/ARK-ASR-0.6B`](https://huggingface.co/Audio8/ARK-ASR-0.6B): distilled 0.6B sibling reusing the ark track (Qwen2.5-0.5B-class decoder), 19 languages, Apache-2.0. Parity pending GPU verification.
- [`mistralai/Voxtral-Mini-4B-Realtime-2602`](https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602): first streaming-native model in the repo: a causal conv embedder + 32-layer sliding-window (750) encoder + ×4-merge GELU projector into a Ministral-3-class decoder (26 layers, GQA, sliding window 8192), with additive audio injection at every position and an AdaRMSNorm delay conditioning (default 6 tokens ≈ 480 ms) on the MLP branch. Decode runs in lockstep with audio positions (12.5 tok/s length cap); per-step encoder slices are fixed-size (4 embeds), the shape a future CUDA-graph decode path captures. The v1 track (`starling.voxtral`) runs an eager loop mirroring stock `generate` exactly (byte-exact by construction; needs the `mistral-common` package for the tekken tokenizer); streaming input and the graphed decode are follow-ups. 13 languages, Apache-2.0. Parity pending GPU verification.

The autoregressive models (granite, moss, qwen3, qwen3_06, ark, ark06) share an
encoder + LLM-decoder pattern where the decode loop is the bottleneck. Parakeet
is a transducer; voxtral decodes
text in lockstep with streaming audio positions.

- [`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini): text-to-text: a 0.6B Qwen3 decoder-only normalizer that rewrites raw ASR transcripts as clean written text (removes fillers, resolves self-corrections, punctuation/truecasing, numbers/emails under a `[Styling|Structure|Context]` control line). No audio front-end: the input embedding is a plain token lookup, so both the CUDA pipeline (`starling.s1.NormalizePipeline`, reusing the qwen3 track's K-step captured decode with a dual-EOS stop) and the in-tree ggml engine (`starling-ggml-s1`, with a C++ BPE encoder + a text path via `starling-serve`'s `POST /normalize`) are byte-exact against stock transformers on all fixture tiers. Apache 2.0 with a naming clause ("S1-mini by Superwhisper").

