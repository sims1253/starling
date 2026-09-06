# Models


Starling supports the following speech recognition models. S1-mini, listed
last, edits transcripts after recognition. See [Python serving](python-serving.md)
and [native serving](native-serving.md) for the models each server accepts.

- [`ibm-granite/granite-speech-4.1-2b`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b): CTC conformer encoder + BLIP2 Q-Former projector + granite-4.0-1b decoder. The in-tree ggml engine (`starling-ggml-granite`, greedy path) runs natively via starling-serve; the Python path adds the optional self-speculative decoding drafting from the encoder's CTC head.
- [`ibm-granite/granite-speech-4.1-2b-nar`](https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar): non-autoregressive. One bidirectional forward: CTC conformer draft + blank slots + bidirectional granite-4.0-1b editor refinement. No decode loop.
- [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3): FastConformer + TDT transducer (no LLM). GPU-side mel + chunking for hour-long audio.
- [`nvidia/parakeet-unified-en-0.6b`](https://huggingface.co/nvidia/parakeet-unified-en-0.6b): Unified FastConformer-RNN-T. NeMo-free port: the `.nemo` checkpoint is loaded directly (no `nemo_toolkit`), the encoder/prediction-net/joint are hand-built in PyTorch, and the encoder + greedy RNN-T decode are captured into CUDA graphs.
- [`OpenMOSS-Team/MOSS-Transcribe-preview-2B`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-preview-2B): Qwen3-omni MoE encoder + Qwen3 decoder.
- [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B): Whisper-style windowed-attention encoder + Qwen3 decoder. The in-tree ggml engine (`starling-ggml-qwen3`, greedy path) runs natively via starling-serve.
- [`AutoArk-AI/ARK-ASR-3B`](https://huggingface.co/AutoArk-AI/ARK-ASR-3B): Whisper encoder + MLP adapter + Qwen2.5 decoder.
- [`CohereLabs/cohere-transcribe-03-2026`](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026): Seq2seq encoder-decoder: 48-layer FastConformer encoder + 8-layer Transformer decoder (self + cross attention).
- [`bosonai/higgs-audio-v3-stt`](https://huggingface.co/bosonai/higgs-audio-v3-stt): Whisper-large-v3 mel + MLP projector + Qwen3-1.7B decoder. The CUDA megakernel (encoder kept eager) runs under its own `.venv-higgs` (transformers 4.51) because the model's `trust_remote_code` modeling breaks under transformers 5.13; the in-tree ggml engine (`starling-ggml-higgs`, full Whisper encoder + avg pool + MLP projector + Qwen3 decoder with qk_norm) runs from the main environment and is byte-exact against the golden on short/medium/long.
- [`nvidia/Nemotron-Labs-Audex-2B`](https://huggingface.co/nvidia/Nemotron-Labs-Audex-2B): Whisper-large-v3 encoder (with avg-pooler) + relu2 projector + Nemotron-Dense 2B decoder (squared-ReLU MLP, not SwiGLU). ASR path only. The in-tree ggml engine (`starling-ggml-audex`, greedy path) runs natively via starling-serve. NVIDIA Oneway Noncommercial License: non-commercial use only, unlike the Apache/MIT-licensed models above.
- [`HojoAI/Hojo-ASR-V1`](https://huggingface.co/HojoAI/Hojo-ASR-V1): Whisper-large-v3 mel + Qwen3-Omni audio tower (3× conv2d + 32 transformer layers) + WeNet Conformer bottleneck (2 blocks, rel-pos MHA + BatchNorm conv module) + Qwen3-4B decoder. Uses beam-4 decoding. The CUDA megakernel runs under `.venv-hojo` (transformers 4.57); the in-tree ggml engine (`starling-ggml-hojo`) runs from the main environment and is byte-exact against the golden on short/medium/long.

The autoregressive models (granite, moss, qwen3, ark, higgs, audex, cohere, hojo) share an
encoder + LLM-decoder pattern where the decode loop is the bottleneck. Parakeet
is a transducer; granite-nar is a single bidirectional pass.

- [`superwhisper/s1-mini`](https://huggingface.co/superwhisper/s1-mini): text-to-text: a 0.6B Qwen3 decoder-only normalizer that rewrites raw ASR transcripts as clean written text (removes fillers, resolves self-corrections, punctuation/truecasing, numbers/emails under a `[Styling|Structure|Context]` control line). No audio front-end: the input embedding is a plain token lookup, so both the CUDA pipeline (`starling.s1.NormalizePipeline`, reusing the qwen3 track's K-step captured decode with a dual-EOS stop) and the in-tree ggml engine (`starling-ggml-s1`, with a C++ BPE encoder + a text path via `starling-serve`'s `POST /normalize`) are byte-exact against stock transformers on all fixture tiers. Apache 2.0 with a naming clause ("S1-mini by Superwhisper").

