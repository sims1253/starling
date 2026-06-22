# GPU mel feature extraction — distilled pipeline (parakeet-tdt-0.6b-v3)

Distilled from `transformers/models/parakeet/feature_extraction_parakeet.py` (285 lines).
The stock `processor(audio)` runs this ENTIRELY on CPU and returns CPU tensors,
which is why feat_ms goes from 68ms (B8) to 974ms (B16) — a CPU throughput cliff.
Moving the pipeline to GPU torch ops eliminates the cliff. Triton fusion is
optional (fuses pre-emphasis + window into the STFT, and fuses CMVN reductions).

## Config
- sampling_rate = 16000
- feature_size (mel bins) = 128
- n_fft = 512
- hop_length = 160  (10ms)
- win_length = 400  (25ms)
- preemphasis = 0.97
- padding_value = 0.0
- LOG_ZERO_GUARD_VALUE = 2**-24 = 5.96e-8
- EPSILON = 1e-5
- mel_filters: torch.Tensor (128, 257), float32, from librosa.filters.mel(sr=16000,
  n_fft=512, n_mels=128, fmin=0, fmax=8000, norm="slaney"). Available as
  `processor.feature_extractor.mel_filters`.
- window: torch.hann_window(400, periodic=False)

## Exact pipeline (8 steps), all tensor ops:

### 1. Pad audio to max length in batch (right-pad with 0.0)
Input: list of 1D float arrays (varying lengths). Output: (B, L_max) float32.

### 2. Pre-emphasis filter (IIR, per-utterance)
```
y[:, 0] = x[:, 0]                          # first sample unchanged
y[:, 1:] = x[:, 1:] - 0.97 * x[:, :-1]     # pre-emphasis
y = y.masked_fill(~timemask, 0.0)          # zero out padding positions
```
where `timemask = arange(L_max)[None,:] < audio_lengths[:,None]`.

### 3. STFT
```python
window = torch.hann_window(400, periodic=False)  # precompute, store on GPU
stft = torch.stft(waveform, n_fft=512, hop_length=160, win_length=400,
                  window=window, return_complex=True, pad_mode="constant")
# stft shape: (B, 257, T_frames)
```

### 4. Magnitude squared (= |STFT|², NOT power=2.0 of abs)
```python
# The reference does this (equivalent to abs(stft)**2):
magnitudes = torch.view_as_real(stft)       # (B, 257, T, 2)
magnitudes = torch.sqrt(magnitudes.pow(2).sum(-1))  # (B, 257, T) = abs(stft)
magnitudes = magnitudes.pow(2)              # (B, 257, T) = abs(stft)**2
```
You can simplify to `magnitudes = stft.abs() ** 2` (same result, fewer ops).

### 5. Mel filterbank
```python
mel_spec = mel_filters @ magnitudes   # (128, 257) @ (B, 257, T) -> (B, 128, T)
# NOTE: mel_filters must broadcast across batch dim. Use mel_filters[None] or einsum.
```

### 6. Log
```python
mel_spec = torch.log(mel_spec + 2**-24)
```

### 7. Permute
```python
mel_spec = mel_spec.permute(0, 2, 1)  # (B, T, 128)
```

### 8. CMVN (per-utterance mean/variance normalization, ignoring padding)
```python
features_lengths = (audio_lengths + n_fft//2*2 - n_fft) // hop_length  # floor div
attention_mask = arange(T)[None, :] < features_lengths[:, None]
mask = attention_mask.unsqueeze(-1)                 # (B, T, 1)
masked = mel_spec * mask
mean = masked.sum(dim=1) / features_lengths.unsqueeze(-1)   # (B, 128)
mean = mean.unsqueeze(1)                                     # (B, 1, 128)
variance = ((masked - mean)**2 * mask).sum(dim=1) / (features_lengths - 1).unsqueeze(-1)
std = torch.sqrt(variance).unsqueeze(1)                     # (B, 1, 128)
mel_spec = (mel_spec - mean) / (std + 1e-5)
mel_spec = mel_spec * mask   # re-zero padding
```

## Output
- `input_features`: (B, T, 128) float32 (cast to bf16 before encoder)
- `attention_mask`: (B, T) bool
These EXACTLY match what the stock `processor(audio)` returns, just computed on GPU.

## Byte-exactness note
- `torch.stft` on GPU vs CPU gives IDENTICAL results (same algorithm).
- The mel_filters matrix is pre-computed from librosa and cached; just copy to GPU.
- `log`, `abs`, matmul, reductions are all deterministic on GPU in float32.
- The whole pipeline in float32 on GPU should match the CPU float32 output to ~1e-6
  (float32 arithmetic ordering differences). For byte-exact match you'd need to
  replicate the exact reduction order, but 1e-6 max-abs is well within the encoder's
  tolerance (the Conformer is robust to feature noise).

## Implementation strategy
1. **Phase 1 (easy win)**: reimplement the 8 steps as pure torch ops on GPU (no Triton).
   Store mel_filters and window as pre-allocated GPU tensors. This alone eliminates
   the CPU bottleneck and H2D transfer. Test: max-abs vs stock < 1e-4.
2. **Phase 2 (optional fusion)**: Triton kernel fusing pre-emphasis+windowing+STFT
   into one pass, and fusing the CMVN reductions. Only if Phase 1 isn't fast enough.
