"""Constants for nvidia/parakeet-unified-en-0.6b.

All values below are **locked from the checkpoint itself** (the
``parakeet-unified-en-0.6b.nemo`` ships a bare ``state_dict`` -- no
``config.yaml`` -- so every dim is read from the tensor shapes, not a config
file). The tokenizer is the sentencepiece model extracted from the eschmidbauer
``parakeet-unified-en-0.6b-c`` weight pack (byte-identical to the original NeMo
tokenizer; cross-checked 0 mismatches against the sherpa-onnx unified export's
``tokens.txt``).

Architecture (locked)
---------------------
* Mel frontend: 128 mels (``preprocessor.featurizer.fb`` is ``(1, 128, 257)``),
  n_fft 512, hop 160, win 400, 16 kHz, hann(periodic=False), preemphasis 0.97,
  mag_power 2, per_feature CMVN. Identical math to ``starling.parakeet.mel_gpu``
  (see that module's ``MEL_PIPELINE.md``).
* Encoder: 24 Conformer layers, d_model 1024, 8 heads (head_dim 128),
  macaron-style (``feed_forward1`` + ``feed_forward2``), relative-pos attention,
  conv module depthwise kernel 9 + BatchNorm1d. Pre-encode:
  Conv2DSubsampling (two strided 3x3 convs -> x4 time) + extra 1x1 convs +
  ``out`` Linear(4096 -> 1024) -> overall x8 subsampling vs mel frames.
* Prediction net: Embedding(1025, 640) + 2-layer LSTM(640 hidden).
* Joint: enc Linear(1024 -> 640) + pred Linear(640 -> 640) -> sum -> ReLU ->
  Linear(640 -> 1025).
* Vocab: 1024 BPE sentencepiece pieces, blank_id 1024 (== vocab_size).

Confirmed against the sherpa-onnx unified non-streaming ONNX export IO:
encoder in ``(B,128,T_mel)`` -> out ``(B,1024,T_enc)``; decoder LSTM state
``(2,B,640)``; joiner out ``(...,1025)``.
"""

from __future__ import annotations

# HuggingFace repo. Only ``parakeet-unified-en-0.6b.nemo`` lives there -- no
# config or tokenizer files ship on the hub; both are inside (or alongside) the
# .nemo and we extract the tokenizer from the eschmidbauer pack instead.
MODEL_ID: str = "nvidia/parakeet-unified-en-0.6b"
NEMO_FILENAME: str = "parakeet-unified-en-0.6b.nemo"

# Tokenizer source: eschmidbauer's weight pack ships the original sentencepiece
# ``tokenizer.model``. We mirror it into the repo's HF cache at first load
# (see loader.py). 1024 BPE pieces, <unk>=0, blank=1024.
TOKENIZER_HF_REPO: str = "eschmidbauer/parakeet-unified-en-0.6b-c"
TOKENIZER_HF_FILE: str = "c_weights_fp32/tokenizer.model"

# --- sample rate / audio ----------------------------------------------------
SAMPLE_RATE: int = 16000

# --- mel frontend (matches starling.parakeet.mel_gpu + this model's fb) -----
N_MELS: int = 128
N_FFT: int = 512
HOP_LENGTH: int = 160        # 10 ms
WIN_LENGTH: int = 400        # 25 ms
PREEMPHASIS: float = 0.97
LOG_ZERO_GUARD_VALUE: float = 2 ** -24
EPSILON: float = 1e-5
PADDING_VALUE: float = 0.0
# per_feature CMVN (mean/var over time, per mel bin) -- NOT global.

# --- encoder ----------------------------------------------------------------
ENCODER_LAYERS: int = 24
ENCODER_D_MODEL: int = 1024
ENCODER_N_HEADS: int = 8
FEED_FORWARD_EXPANSION: int = 4     # linear1: 1024 -> 4096
CONV_KERNEL: int = 9                # depthwise conv kernel
SUBSAMPLING_FACTOR: int = 8         # mel frames -> encoder frames
# 2 strided 3x3 convs -> x4 on time; the pre-encode `out` Linear collapses the
# (256, T/4, 4) channel block into 1024. Overall x8 (featurizer hops already
# accounted for separately by the mel extractor).

# --- RNNT decoder / joint ---------------------------------------------------
VOCAB_SIZE: int = 1024              # BPE pieces (sentencepiece)
BLANK_ID: int = 1024                # == VOCAB_SIZE; <blk> appended after the spm vocab
NUM_TOKENS_WITH_BLANK: int = 1025   # joint output dim
PRED_HIDDEN: int = 640
PRED_RNN_LAYERS: int = 2
JOINT_HIDDEN: int = 640
MAX_SYMBOLS_PER_STEP: int = 10      # RNNT guard (NeMo default; verified from
                                    # istupakov sibling config "max_tokens_per_step":10)

# --- derived ----------------------------------------------------------------
# mel frames per second = SAMPLE_RATE / HOP_LENGTH = 100
# encoder frames per second = 100 / SUBSAMPLING_FACTOR ~= 12.5
SAMPLES_PER_ENC_FRAME: int = HOP_LENGTH * SUBSAMPLING_FACTOR   # 1280
__all__ = [name for name in dir() if name.isupper()]
