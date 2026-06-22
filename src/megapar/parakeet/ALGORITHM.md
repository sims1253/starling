# TDT decode algorithm — distilled reference (parakeet-tdt-0.6b-v3)

Distilled by the orchestrator from the transformers source so workers don't have
to crawl the 1154-line modeling file. Every shape and branch below is VERIFIED
against:
- `transformers/models/parakeet/generation_parakeet.py` (ParakeetTDTGenerationMixin)
- `transformers/models/parakeet/modeling_parakeet.py` (ParakeetForTDT, decoder, joint)

## Config values (parakeet-tdt-0.6b-v3)
- vocab_size = 8193
- blank_token_id = 8192  (== decoder_start_token_id; blank doubles as the start token)
- durations = [0, 1, 2, 3, 4]  (5 duration classes; the index maps to the value)
- num_decoder_layers = 2
- decoder_hidden_size = 640
- encoder_hidden_size = 1024  (projected to 640 by encoder_projector)
- max_symbols_per_step = 10   (RNN-T guard; NOT used in TDT path, only RNNT)
- joint head output dim = vocab_size + len(durations) = 8198

## Submodule paths on the loaded ParakeetForTDT
- `model.encoder` — ParakeetEncoder (the 24-layer Conformer)
- `model.encoder_projector` — nn.Linear(1024 -> 640)
- `model.decoder` — ParakeetRNNTDecoder { embedding: nn.Embedding(8193,640),
  lstm: nn.LSTM(640,640,num_layers=2,batch_first=True),
  decoder_projector: nn.Linear(640,640), blank_token_id=8192 }
- `model.joint` — ParakeetTDTJointNetwork { activation: ACT2FN[hidden_act],
  head: nn.Linear(640, 8198) }   # 8198 = 8193 tokens + 5 durations

## Encoder feature path (precompute ONCE per utterance)
```python
enc = model.get_audio_features(input_features, attention_mask)
# enc.pooler_output shape: (B, T_enc, 640)  -- already projected 1024->640
# enc.attention_mask shape: (B, T_enc) bool -- True = valid frame
# valid_lengths = enc.attention_mask.sum(-1)  -- per-utterance valid frame count
```
`get_audio_features` runs `self.encoder(...)` then `pooler_output = encoder_projector(last_hidden_state)`.

## Per-step decode math (the thing you graph-capture)

State (all device-side, static buffers for graph capture):
- `frame_idx`: LongTensor (B,) — current encoder frame pointer per batch element
- `last_token`: LongTensor (B,) — last emitted token (init = blank = 8192)
- LSTM cache: `cache`, `hidden_state` (2,B,640), `cell_state` (2,B,640)
  — the model's own `ParakeetRNNTDecoderCache` already does `mark_static_address`
  in `lazy_initialization`, so reuse it.

One decode step:
```python
# 1. gather the current encoder frame (clamp guards against reading past end)
max_len = pooler_output.shape[1]
idx = frame_idx.clamp(max=max_len - 1)           # (B,)
enc_frame = pooler_output[torch.arange(B), idx]   # (B, 640)

# 2. run the decoder on the last token, with blank-skip freeze built in.
#    decoder_input_ids must be shape (B, 1).
decoder_input = last_token.unsqueeze(1)           # (B, 1)
decoder_out = model.decoder(decoder_input, cache=decoder_cache)  # (B, 1, 640)
# NOTE: the decoder's forward ALREADY implements blank-skip: if
# cache.is_initialized and last_token is blank for ALL batch elements, it returns
# cache.cache (frozen) WITHOUT running the LSTM. For batched decode where SOME
# elements are blank and others aren't, it runs the LSTM and uses `torch.where`
# with `mask = ~blank_mask` to keep frozen state for blank elements.

# 3. joint: ReLU(enc_frame + decoder_out) -> head -> logits
logits = model.joint(
    encoder_hidden_states=enc_frame[:, None, None, :],     # (B, 1, 1, 640)
    decoder_hidden_states=decoder_out[:, None, :, :],      # (B, 1, 1, 640)
).squeeze(2)                                                # (B, 1, 8198)
logits = logits[:, -1, :]                                   # (B, 8198)

# 4. greedy pick token + duration index
token = logits[:, :8193].argmax(dim=-1)            # (B,)
dur_idx = logits[:, 8193:].argmax(dim=-1)          # (B,) in [0,4]
dur = torch.tensor([0,1,2,3,4], device=...)[dur_idx]  # (B,) actual frame advance

# 5. TDT frame-advance rule (DIFFERENT from RNN-T!):
#    blank AND dur==0  ->  force dur = 1  (guarantee forward progress)
#    otherwise        ->  dur as predicted
blank_mask = token == 8192
dur = torch.where(blank_mask & (dur == 0), torch.ones_like(dur), dur)
frame_idx = frame_idx + dur

# 6. emitted token for THIS step is `token` (include blanks in the sequence —
#    the processor.decode skips them via skip_special_tokens=True).
#    last_token for NEXT step = token.
# 7. stop when frame_idx >= valid_lengths  (per batch element; track a finished mask).
```

## Why this is graph-friendly
- Everything in steps 1-5 is tensor ops on static-shape, static-address buffers.
- The only host-side concerns are: the output-length cap (sized via
  `_prepare_generated_length` = max_symbols_per_step * T_enc, but TDT advances
  by duration so the real max is ~ sum of max durations * T_enc; size generously
  and stop early via the finished mask), and appending emitted tokens to the
  output buffer.
- The decoder cache buffers (hidden_state, cell_state, cache) are mutated in
  place by `cache.update(...)`; with `mark_static_address` they survive replay.
- `frame_idx` and `last_token` must be YOUR static buffers (pre-allocated,
  marked static), advanced in place, so the graph reads their current value on
  replay.

## Output handling (matches stock generate)
- The stock path prepends `decoder_start_token_id` (blank) to the sequence, so
  your output buffer should start with [blank] and append one token per step.
- `processor.batch_decode(seqs, skip_special_tokens=True)` strips blanks and
  special tokens and joins remaining piece ids into text. Your emitted sequence
  (with blanks included) must match `oracle.json`'s token sequence when decoded.
- Return shape: (B, T_out) padded with pad_id (processor.tokenizer.pad_token_id).

## Correctness oracle
`outputs/oracle.json` has the deterministic greedy transcript text for the
short/medium/long fixtures. Your decode, fed through
`processor.batch_decode(..., skip_special_tokens=True)`, must equal the oracle
text BYTE-FOR-BYTE. (Greedy TDT decode is deterministic.)
