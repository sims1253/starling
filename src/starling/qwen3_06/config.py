"""Architecture constants for the Qwen3-ASR-0.6B megakernel pipeline.

Everything the fused kernels, CUDA-graph capture, and correctness checks need
to size buffers, decode shapes, or compare against the eager reference lives
here so there is a single source of truth for the Qwen/Qwen3-ASR-0.6B-hf
architecture: a windowed-attention conv encoder feeding a 2-layer MLP
projector that maps the 896-wide encoder output into the 1024-wide Qwen3
decoder hidden size.

Reuse contract with the 1.7B track: the ``-hf`` checkpoint ships a
byte-identical ``tokenizer.json`` / ``tokenizer_config.json`` /
``chat_template.jinja`` / ``processor_config.json`` / ``generation_config.json``
(verified against revision 7f1569a4), identical special-token ids
(``audio_token_id`` 151676, ``pad``/``eos`` 151645), the same whisper-style
128-bin mel frontend (n_fft 400, hop 160, n_window 50, n_window_infer 800,
max_pos_emb 13), and the same windowed-attention encoder / MLP-projector /
Qwen3-trunk topology. Only the dims below differ: the encoder narrows to
896 hidden over 18 layers / 14 heads / 3584 FFN, and the decoder trunk runs
1024 hidden with a 3072-wide SwiGLU while KEEPING the 1.7B's 16Q/8KV GQA at
head_dim 128 — so the attention width (2048) is wider than the trunk (1024).
The megakernels derive every dim from the loaded tensors / model config, so
the qwen3 track's encoder/LLM/pipeline modules are reused directly.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Model identity
# ---------------------------------------------------------------------------
MODEL_ID: str = "Qwen/Qwen3-ASR-0.6B-hf"
"""HF hub repo id for the transformers-native (``-hf``) Qwen3-ASR 0.6B variant."""

# ---------------------------------------------------------------------------
# Audio encoder dims (Qwen3ASREncoderConfig / audio_config)
# ---------------------------------------------------------------------------
AUDIO_D_MODEL: int = 896             # encoder hidden
AUDIO_NUM_LAYERS: int = 18           # encoder layers
AUDIO_NUM_HEADS: int = 14            # encoder_attention_heads
AUDIO_HEAD_DIM: int = 64             # d_model // num_heads (896 // 14)
AUDIO_FFN_DIM: int = 3584            # encoder_ffn_dim
AUDIO_NUM_KV_HEADS: int = 14         # MHA in the audio tower (num_key_value_heads)
AUDIO_NUM_MEL_BINS: int = 128        # mel bins
AUDIO_DOWNSAMPLE_HIDDEN: int = 480   # conv2d channel dim
AUDIO_OUTPUT_DIM: int = 1024         # projector output dim (== LLM hidden)
AUDIO_N_WINDOW: int = 50             # chunk size is n_window*2 = 100 frames
AUDIO_N_WINDOW_INFER: int = 800      # inference attention window (raw frames)
AUDIO_CONV_CHUNKSIZE: int = 500      # conv_chunksize (unused at infer, kept for ref)
AUDIO_MAX_POS_EMB: int = 13          # max_position_embeddings (post-CNN frames)

# ---------------------------------------------------------------------------
# Text decoder dims (Qwen3 text_config)
# ---------------------------------------------------------------------------
LLM_HIDDEN_SIZE: int = 1024
LLM_NUM_LAYERS: int = 28
LLM_NUM_ATTN_HEADS: int = 16
LLM_NUM_KV_HEADS: int = 8            # GQA (16 Q / 8 KV)
LLM_HEAD_DIM: int = 128              # explicit in config; 16*128 = 2048 != 1024
LLM_INTERMEDIATE_SIZE: int = 3072
LLM_VOCAB_SIZE: int = 151936
LLM_MAX_POS_EMB: int = 65536
LLM_ROPE_THETA: float = 1_000_000.0
LLM_RMS_NORM_EPS: float = 1e-6

# ---------------------------------------------------------------------------
# Tokenisation / multimodal (identical to the 1.7B track)
# ---------------------------------------------------------------------------
AUDIO_TOKEN_ID: int = 151676
"""The ``<|audio|>`` placeholder token id; positions carrying it are clobbered
by the projected audio embeddings inside ``Qwen3ASRModel.forward``."""
EOS_TOKEN_ID: int = 151645           # im_end (primary EOS for greedy stop)
PAD_TOKEN_ID: int = 151645

DEFAULT_TASK_PROMPT: str = ""
"""Qwen3-ASR takes no instruction prompt: the chat template just wraps the
``<|audio|>`` placeholder. ``build_inputs`` leaves the user content empty."""
