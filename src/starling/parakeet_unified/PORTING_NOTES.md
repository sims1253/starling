# Porting notes — parakeet-unified-en-0.6b (NeMo-free megakernel)

This module ports `nvidia/parakeet-unified-en-0.6b` (Unified FastConformer-RNN-T)
as a starling megakernel **without any NeMo dependency**. The `.nemo` checkpoint
is loaded directly, the three networks are hand-built in PyTorch, and the
encoder + greedy RNN-T decode are captured into CUDA graphs (mirroring the
sibling `starling.parakeet` TDT pipeline).

## Why NeMo-free

NeMo's runtime conflicts with this repo's pinned `torch>=2.11` (cu130); the
`nemo_toolkit` install pulls a different torch and breaks every other model.
sherpa-onnx (the other natural reference) lags torch 2.11/cu130 too. So the
port loads the checkpoint directly and uses the hand-built eager greedy RNN-T
loop as its byte-exact reference (the loop mirrors NeMo's
`rnnt_greedy_decoding` algorithm exactly).

## Checkpoint loading (`loader.py`)

The HF repo ships a single `parakeet-unified-en-0.6b.nemo` (2.47 GB) which is a
zip containing **only** a torch flat-checkpoint under `model_weights/` — no
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
  mag_power 2, per_feature CMVN. Identical math to `starling.parakeet.mel_gpu`.
* Encoder: 24 Conformer layers, d_model 1024, 8 heads (head_dim 128),
  macaron-style (`feed_forward1` + `feed_forward2`), relative-pos attention,
  conv module depthwise kernel 9 + BatchNorm1d. Pre-encode: 3 stride-2 conv
  blocks (pointwise → depthwise → pointwise mix → ReLU) → overall ×8
  subsampling vs mel frames; the flattened `(256, 16) = 4096` channel block is
  collapsed by the `out` Linear(4096 → 1024).
* Prediction net: `Embedding(1025, 640)` + 2-layer `LSTM(640)`.
* Joint: `Linear(1024→640)` + `Linear(640→640)` → sum → ReLU →
  `Linear(640→1025)`. The checkpoint's `joint_net.2.*` is the only Linear in
  the `joint_net` Sequential (indices 0/1 are Identity/ReLU with no params);
  `_JointNet._load_from_state_dict` remaps `joint_net.2.* → linear.*`.
* Vocab: 1024 BPE sentencepiece pieces, blank_id 1024 (== vocab_size, appended
  after the spm vocab by NeMo's RNNT decoding; NOT a sentencepiece piece).

## Weight-key remapping (`modeling.py`)

The encoder's pre-encode conv uses non-contiguous ModuleList indices in the
checkpoint (`conv.0, conv.2, conv.3, conv.5, conv.6` — `conv.1/4/7` are
weightless ReLUs). `ConformerEncoder.load_state_dict_prefixed` remaps them to
the contiguous `pre_encode_conv.conv.0..4` via an explicit index map. All other
key paths (`encoder.layers.N.*`, `decoder.prediction.*`, `joint.*`) match the
hand-built modules verbatim, so `load_state_dict(strict=True)` is the
byte-exact gate (decoder + joint load strict with zero remapping).

## The RNN-T megakernel (`decode_mega.py`)

The TDT megakernel (`starling.parakeet.decode_mega`) captures K decode steps
per replay with an in-graph blank-skip freeze. The RNN-T port keeps the
scaffolding (static buffers, K-step capture, ring buffer +
single-sync-per-replay, in-graph `last_token` chaining, device-side
`torch.where` branching) but rewrites `_step_fn` for RNN-T:

* **No eager step-0 prefill.** The RNN-T prediction net has no host-side
  cache-init branch (we drive `nn.LSTM` directly with explicit state tensors),
  so step 0 is capture-safe and runs inside the graph.
* **The LSTM always advances.** Standard RNN-T runs the prediction net on every
  step (including after a blank); the TDT blank-skip freeze does NOT apply.
* **`last_token` is NOT reset on blank.** On a real emission
  `last_token <- tok`; on blank `last_token` is LEFT UNCHANGED (the eager
  oracle's `decode_eager.greedy_decode` only updates `last_token` on a real
  emission — the next step's prediction net re-runs on the same last emitted
  token, NOT on blank). This was the subtle correctness bug: resetting
  `last_token` to blank on a blank emission diverged from the eager oracle.
* **`sym_count` device-side cap.** `max_symbols_per_step` (10) is enforced
  device-side: `sym_count >= max_symbols` forces a blank, advancing the frame
  and resetting the counter — byte-exact with the eager
  `while not_blank and symbols < max_symbols` guard.
* **Finished rows freeze.** `frame_idx >= valid_lengths` → `last_token <- blank`
  so finished rows keep emitting blank + advancing past the end until the host
  stops the loop.

## Correctness gates

Golden capture: `scripts/parakeet_unified_golden.py` runs the eager port on
`tests/fixtures/{short,medium,long}.wav` and persists token ids + text to
`golden/parakeet_unified_*`. Tests (`tests/test_parakeet_unified_pipeline.py`):

1. eager greedy decode reproduces the golden token sequence (self-consistency
   regression guard),
2. graphed encoder is byte-exact with eager (max_diff 0.0),
3. the RNN-T megakernel reproduces the eager greedy token sequence across
   K ∈ {1, 4, 16, 64},
4. the integrated pipeline `transcribe` matches the golden transcript,
5. single-chunk chunker is byte-exact with the one-shot pipeline path.

All gates pass at fp32. The bf16 path is byte-identical to the bf16 eager path
(the standard sub-ULP Conformer rounding flips an occasional argmax vs fp32).

## Chunking (`chunking.py`)

RNN-T has no per-token duration (unlike TDT), but the decoder's running
`frame_idx` — the encoder frame the decoder is currently consuming — is still
the absolute encoder-frame position of each emitted token. The chunker recovers
that per-token cumulative frame index and converts chunk-local frame indices to
global sample positions (`chunk_start + frame * 1280`), then left-biased dedups
overlap regions exactly like the TDT chunker.
