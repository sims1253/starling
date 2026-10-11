# Porting notes: parakeet-unified-en-0.6b (NeMo-free reference)

This package is a hand-built PyTorch port of `nvidia/parakeet-unified-en-0.6b`
(Unified FastConformer-RNN-T) that runs **without NeMo**. It is the reference
for the native port tracked in issue #358: `scripts/goldens/parakeet_unified_golden.py`
uses it to capture `golden/parakeet_unified_*` without `nemo_toolkit`.

It was the eager oracle of the removed Python/CUDA research pipeline (CUDA-graph
encoder and RNN-T megakernel, chunker, and their tests). That code is at tag
`python-research-final` under `src/starling/parakeet_unified/`.

## Why NeMo-free

NeMo's runtime conflicts with recent torch; the `nemo_toolkit` install pulls a
different torch. sherpa-onnx (the other natural reference) lagged too. So the
port loads the checkpoint directly and uses a hand-built eager greedy RNN-T
loop as its byte-exact reference (the loop mirrors NeMo's
`rnnt_greedy_decoding` algorithm).

## Checkpoint loading (`loader.py`)

The HF repo ships a single `parakeet-unified-en-0.6b.nemo` (2.47 GB) which is a
zip containing **only** a torch flat-checkpoint under `model_weights/`: no
`config.yaml`, no tokenizer. So:

* weights: copy the `model_weights/` entries out of the `.nemo` into a
  standalone `model_weights.zip` (preserving the single top-level subdir that
  PyTorchFileReader requires) once, cache it under
  `~/.cache/starling/parakeet_unified/`, and `torch.load` that. Returns 989
  keys, no `state_dict` wrapper, no `model.` prefix.
* tokenizer: the sentencepiece model from
  `eschmidbauer/parakeet-unified-en-0.6b-c` (byte-identical to the original;
  cross-checked against the sherpa-onnx unified export's `tokens.txt`).

## Dims (locked from the tensor shapes, in `config.py`)

The checkpoint ships a bare `state_dict` (no config), so every dim is read from
the tensor shapes:

* Mel frontend: 128 mels (`preprocessor.featurizer.fb` is `(1, 128, 257)`),
  n_fft 512, hop 160, win 400, 16 kHz, hann(periodic=False), preemphasis 0.97,
  mag_power 2, per_feature CMVN. Same NeMo math as the parakeet-tdt frontend.
* Encoder: 24 Conformer layers, d_model 1024, 8 heads (head_dim 128),
  macaron-style (`feed_forward1` + `feed_forward2`), relative-pos attention,
  conv module depthwise kernel 9 + BatchNorm1d. Pre-encode: 3 stride-2 conv
  blocks (pointwise, depthwise, pointwise mix, ReLU), overall x8
  subsampling vs mel frames; the flattened `(256, 16) = 4096` channel block is
  collapsed by the `out` Linear(4096 -> 1024).
* Prediction net: `Embedding(1025, 640)` + 2-layer `LSTM(640)`.
* Joint: `Linear(1024->640)` + `Linear(640->640)`, sum, ReLU,
  `Linear(640->1025)`. The checkpoint's `joint_net.2.*` is the only Linear in
  the `joint_net` Sequential (indices 0/1 are Identity/ReLU with no params);
  `_JointNet._load_from_state_dict` remaps `joint_net.2.* -> linear.*`.
* Vocab: 1024 BPE sentencepiece pieces, blank_id 1024 (== vocab_size, appended
  after the spm vocab by NeMo's RNNT decoding; NOT a sentencepiece piece).

## Weight-key remapping (`modeling.py`)

The encoder's pre-encode conv uses non-contiguous ModuleList indices in the
checkpoint (`conv.0, conv.2, conv.3, conv.5, conv.6`; `conv.1/4/7` are
weightless ReLUs). `ConformerEncoder.load_state_dict_prefixed` remaps them to
the contiguous `pre_encode_conv.conv.0..4` via an explicit index map. All other
key paths (`encoder.layers.N.*`, `decoder.prediction.*`, `joint.*`) match the
hand-built modules verbatim, so `load_state_dict(strict=True)` is the
byte-exact gate (decoder + joint load strict with zero remapping).

## Greedy RNN-T semantics (`decode_eager.py`)

A native port must match these details of the reference loop:

* **The LSTM always advances.** Standard RNN-T runs the prediction net on every
  step, including after a blank (no TDT-style blank-skip freeze).
* **`last_token` is NOT reset on blank.** On a real emission
  `last_token <- tok`; on blank it is left unchanged, so the next step's
  prediction net re-runs on the same last emitted token.
* **`max_symbols_per_step` (10)** caps emissions per encoder frame: reaching it
  advances to the next frame as a blank would.

The golden is captured in fp32. A bf16 run is self-consistent but the usual
sub-ULP Conformer rounding can flip an occasional argmax against fp32.
