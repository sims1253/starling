# MOSS-Transcribe-preview-2B → Starling ggml implementation contract

**Status:** authoritative implementation specification for the batch-1 transcription path.  
**Reference checkpoint:** `OpenMOSS-Team/MOSS-Transcribe-preview-2B`, snapshot `c98175cb20e48bd9be4e95f6c85f2af18899f780`.  
**Correctness target:** the newly generated greedy token IDs, and then tokenizer-decoded UTF-8 text, must equal `golden/moss_{short,medium,long}_{ids.pt,text.txt}` exactly. Tensor equality is not the public contract, but every arithmetic detail below should be treated as fixed until an end-to-end token probe proves a substitution harmless.

> **Normative-path warning.** The oracle is not Hugging Face `generate()`. It is `scripts/moss_golden.py` calling `audio_features`, `build_inputs_embeds`, and the manual `greedy_generate` loop in `src/starling/moss/reference.py` (`scripts/moss_golden.py:36-69`; `src/starling/moss/reference.py:23-112`). The model is loaded as BF16 with eager attention (`src/starling/moss/loader.py:34-58`). Optimized Python modules are corroborating implementations, not the definition of correctness.

## 1. Model configuration

Values in the checkpoint's nested `audio_config` and `language_config` are authoritative; do **not** use similarly named top-level composite-config values. The latter describe the `MossConfig` parent and include irrelevant values such as top-level hidden size 4096 and RoPE theta 10000 (`HF snapshot config.json:29-35,125-139`).

| Subsystem | Parameter | Value | Normative source |
|---|---:|---:|---|
| frontend | sample rate | 16,000 Hz | `src/starling/moss/vendor/processing_Moss.py:11-17` |
| frontend | mel bins | 128 | loader overrides only `mel_dim` at `src/starling/moss/loader.py:60-63`; encoder config `HF config.json:25` |
| frontend | FFT/window length | **640** | `MelConfig` default and loader override behavior: `processing_Moss.py:11-17`, `loader.py:62`; see §2 |
| frontend | hop | 160 | `processing_Moss.py:11-17` |
| frontend | output dtype | BF16 | `processing_Moss.py:17,189` |
| encoder | layers / d_model | 32 / 1280 | `HF config.json:13,18` |
| encoder | heads / head_dim | 20 / 64 | `HF config.json:16`; division enforced at `modeling_qwen3_omni_moe.py:503-517` |
| encoder | FFN width | 5120 | `HF config.json:17` |
| encoder | activation / dropout | exact GELU / all dropout 0 | `HF config.json:9-11,15`; activation mapping and calls at `modeling_qwen3_omni_moe.py:594-605,629-631` |
| encoder | Conv2d stack | channels 1→480→480→480; kernel 3×3; stride 2×2; padding 1×1; bias true | `modeling_qwen3_omni_moe.py:765-767`; width `HF config.json:14` |
| encoder | conv output frequency | 128→64→32→16; flattened width 480×16=7680 | construction formula `modeling_qwen3_omni_moe.py:768-772` |
| encoder | conv_out | 7680→1280, no bias | same source |
| encoder | chunk size | `2*n_window=100` raw mel frames | `HF config.json:22`; `modeling_qwen3_omni_moe.py:643-675` |
| encoder | inference attention window | 800 raw frames = 8 full 100-frame chunks = normally 104 post-CNN tokens | `HF config.json:23`; exact derivation `modeling_qwen3_omni_moe.py:721-735` |
| encoder | conv batch chunk | at most 500 **audio chunks** per Conv2d invocation; no semantic boundary | `HF config.json:12`; `modeling_qwen3_omni_moe.py:800-807` |
| encoder | max sinusoid positions | 1500 | `HF config.json:20`; table construction `modeling_qwen3_omni_moe.py:92-110` |
| encoder | LayerNorm epsilon | PyTorch default 1e-5 | constructors omit `eps`: `modeling_qwen3_omni_moe.py:599,605,764`; PyTorch `nn.LayerNorm` default |
| encoder | post head | LN(1280), Linear 1280→1280+bias, exact GELU, Linear 1280→2048+bias | `modeling_qwen3_omni_moe.py:764,773-775,824-828` |
| adapter | dimensions | 2048→8192 gated→2048 | `HF config.json:2`; `modeling_Moss.py:115-119` |
| decoder | layers / hidden | 28 / 2048 | `HF config.json:47,84` |
| decoder | Q heads / KV heads / head_dim | 16 / 8 / 128 (GQA repeat=2) | `HF config.json:45,83-85` |
| decoder | SwiGLU intermediate | 6144, SiLU | `HF config.json:46,49`; `modeling_qwen3.py:70-83` |
| decoder | projections | Q/K/V/O and gate/up/down all bias-free | `HF config.json:40`; `modeling_qwen3.py:236-250,70-79` |
| decoder | RMSNorm epsilon | 1e-6 | `HF config.json:86`; implementation `modeling_qwen3.py:49-67` |
| decoder | RoPE | default, non-interleaved rotate-half; theta 1,000,000; no scaling | `HF config.json:87-89`; `modeling_qwen3.py:105-180` |
| decoder | context metadata | 40,960 positions; all 28 layers full attention, no sliding window | `HF config.json:50-93` |
| decoder | vocab / tied output | 151,936 / tied | `HF config.json:90,93`; explicit tying `modeling_Moss.py:188-207` |
| runtime | checkpoint/activation dtype | BF16 model and mel; norms and softmax internally F32 as specified below | `HF config.json:29,43`; `loader.py:34-58`; `processing_Moss.py:17` |

The audio encoder class name contains “Moe” only because it is vendored from Qwen3-Omni-MoE. Its audio layers are ordinary dense attention + dense FFN (`modeling_qwen3_omni_moe.py:500-640`). MOSS also instantiates an ordinary dense `Qwen3Model`, not the Omni thinker (`modeling_Moss.py:106-119`). **There is no router, expert selection, or MoE operation anywhere in this transcription path.**

## 2. Audio frontend

### 2.1 Normative waveform and mel algorithm

Golden capture reads with `soundfile`, resamples non-16-kHz files with librosa, averages channels, and casts to float32 (`scripts/moss_golden.py:22-33`). The C API should instead require mono 16-kHz float32 exactly, consistent with `cpp/include/starling_ggml.h:72-80`; resampling is outside the model contract.

`MossProcessor.__call__` converts input to NumPy float32, takes row 0 (not a channel mean) if a 2-D array reaches it, and calls the **private NumPy** Whisper extractor directly (`processing_Moss.py:167-189`). Consequently the public Whisper extractor's 30-second padding/truncation pipeline is bypassed.

For mono float32 samples `x[0..L-1]`:

1. Use `n_fft = frame_length = 640`, hop 160, one-sided RFFT (321 bins), periodic Hann returned by `window_function(640, "hann")`, no dither and no pre-emphasis (`processing_Moss.py:11-17,47-53`; `feature_extraction_whisper.py:105-127`; spectrogram defaults at `audio_utils.py:790-809`).
2. Center frames by reflect-padding 320 samples on each side (`audio_utils.py:934-944`). NumPy promotes waveform and window to float64; each RFFT result is stored as complex64 (`audio_utils.py:939-951,969`). For frontend numerical equivalence, reproduce this mixed-precision sequence rather than assuming an f32 FFT is identical.
3. Compute `abs(RFFT)^2` through the NumPy path, then transpose (`audio_utils.py:972-976`).
4. Apply the 321×128 Slaney mel bank: 0–8000 Hz, `norm="slaney"`, `mel_scale="slaney"` (`feature_extraction_whisper.py:95-103`). Floor the matrix product at `1e-10` (`audio_utils.py:978-979`).
5. Take base-10 logarithm and cast the spectrogram to float32 (`feature_extraction_whisper.py:126`; `audio_utils.py:981-996`).
6. Drop the final time frame (`feature_extraction_whisper.py:128`). Thus `T = floor(L/160)` for the fixture lengths.
7. Global dynamic-range clamp: `m = max(log_spec)` over **all bins and frames**, then `log_spec = max(log_spec, m-8.0)` elementwise; normalize `(log_spec+4.0)/4.0` (`feature_extraction_whisper.py:128-130`). There is no mean/variance normalization.
8. Convert the resulting `[128,T]` float32 array to BF16 before encoder transfer (`processing_Moss.py:189-194`).

### 2.2 Resolving 400 vs 640

**The existing PyTorch goldens use 640, not 400.** `MelConfig.mel_n_fft` defaults to 640 (`processing_Moss.py:11-17`), and `load_model_and_processor` constructs `MelConfig(mel_dim=128)` without changing that field (`loader.py:60-63`). The private extractor receives this value explicitly (`processing_Moss.py:47-53`).

The read-only cstr GGUF `/home/m0hawk/asr-bench/models/moss-transcribe-preview-2b-f16.gguf` instead contains `audio.mel_filters [128,201]` and `audio.mel_window [400]`; those imply FFT 400. That conversion does **not** describe Starling's golden path and must not be copied for byte-exact reproduction. Starling's GGUF must store a 321×128 mel filter tensor and length-640 window (or regenerate exactly). `src/starling/moss/config.py:78-81` correctly says 640; the cstr file is the deviation.

### 2.3 Lengths, chunking, and positions

Define the module-level/processor “deepstack” function for nonnegative integer mel length `T`:

```text
r  = T % 100
l1 = floor((r - 1)/2) + 1
A(T) = floor((floor((l1 - 1)/2)+1 - 1)/2) + 1 + 13*floor(T/100)
```

Use floor division with Python semantics. For positive tails this is three stride-2 same-padding reductions; each full 100-frame chunk contributes 13. This function is independently present in the processor (`processing_Moss.py:78-85`) and at module scope (`modeling_qwen3_omni_moe.py:149-157`). The processor uses it to choose placeholder count (`processing_Moss.py:193-199`). Encoder `get_valid_indices` and `get_audio_cu_seqlens` use the module-level version (`modeling_qwen3_omni_moe.py:678-735`).

There is a second, misleading **instance method** `Qwen3OmniMoeAudioEncoder._get_feat_extract_output_lengths`: it returns a pair after only two reductions (`modeling_qwen3_omni_moe.py:831-838`). The current forward never calls it. Do not use it in the port.

The encoder does chunk its **mel tensor**, but not into independent 30-second model calls:

* For each sample, split the concatenated mel stream into ordered 100-frame pieces; the tail has `T mod 100` frames, with zero remainder interpreted as 100. Right-pad all pieces to that batch's longest piece (`modeling_qwen3_omni_moe.py:643-675`).
* Run all pieces through the three convs. The `conv_chunksize=500` split merely limits how many pieces are convolved at once and concatenates results; it changes neither receptive fields nor attention (`modeling_qwen3_omni_moe.py:800-808`). Conv receptive fields never cross a 100-frame piece boundary.
* Flatten chunk-major padded conv output, then select only valid post-CNN positions using `valid_indices`; ordering remains sample-major, chunk-major, time-major (`modeling_qwen3_omni_moe.py:690-693,809-818`).
* There is **no 30-second truncation or windowing** because `__call__` directly invokes `_np_extract_fbank_features` rather than `WhisperFeatureExtractor.__call__` (`processing_Moss.py:183-189`). The 74.35-second fixture reaches the encoder as 7435 frames and 967 audio tokens.

Fixture constants, derivable without running the model:

| fixture | samples | mel `T` | placeholders `A(T)` | prompt length `A+10` |
|---|---:|---:|---:|---:|
| short | 118,960 | 743 | 97 | 107 |
| medium | 356,880 | 2,230 | 290 | 300 |
| long | 1,189,600 | 7,435 | 967 | 977 |

`max_source_positions=1500` sizes the sinusoid table **per padded chunk**. Current chunks are at most 100 raw frames and 13 post-CNN positions, so long packed streams can exceed 1500 total tokens safely; only the local padded time dimension indexes the table (`modeling_qwen3_omni_moe.py:760-764,812-817`). The practical long-audio limit is the LLM prompt/cache: golden uses cache 2048 (`scripts/moss_golden.py:61`).

## 3. Audio encoder, operation by operation

All weights are loaded to BF16 by `torch_dtype=torch.bfloat16`; model is eval and all configured dropout is zero (`loader.py:53-58`; `HF config.json:9-15`).

### 3.1 Subsampling

Input after chunking is `[C,128,P]`, where `C=ceil(T/100)` for batch 1 and `P=100` except an utterance shorter than 100 may make `P=T`. Insert channel dimension to `[C,1,128,P]` (`modeling_qwen3_omni_moe.py:797-800`). For each Conv2d:

```text
y1 = GELU_exact(conv2d(x, W1, b1, kernel=3x3, stride=2x2, padding=1x1))
y2 = GELU_exact(conv2d(y1,W2,b2, same params))
y3 = GELU_exact(conv2d(y2,W3,b3, same params))
```

The call is `torch.nn.functional.gelu` with default `approximate="none"`, i.e. erf GELU, not tanh (`modeling_qwen3_omni_moe.py:803-806`). Encoder FFNs and projection use config activation `"gelu"`, mapped to `GELUActivation`→`nn.functional.gelu`, also exact/erf (`transformers/activations.py:65-86,325`; `modeling_qwen3_omni_moe.py:601,774`).

For a full chunk, shape evolves frequency/time `128×100 → 64×50 → 32×25 → 16×13`. Permute output `[C,480,16,13] → [C,13,480,16]`, contiguous-flatten to `[C,13,7680]`, then bias-free linear to `[C,13,1280]` (`modeling_qwen3_omni_moe.py:809-810`).

### 3.2 Position and packing

Build the nonpersistent F32 table once:

```text
H = 1280; K = H/2 = 640; max_timescale = 10000
inc = ln(10000)/(K-1)
inv[k] = exp(-inc*k), k=0..639
angle[p,k] = p*inv[k]
pos[p,:] = concat(sin(angle[p,:]), cos(angle[p,:]))
```

(`modeling_qwen3_omni_moe.py:92-107`). Slice the first padded post-conv length positions, cast to activation dtype BF16, and add to every chunk before removing padded rows (`modeling_qwen3_omni_moe.py:812-818`). `scale_embedding=false`, and the computed `embed_scale` is not applied (`HF config.json:27`; forward lines above).

For each chunk length `c_i`, compute `a_i=A(c_i)`. Let `M=max_i a_i`. `valid_indices` are flattened row-major indices `(i*M+t)` for all `0<=t<a_i`; index-select creates packed `[sum a_i,1280]` (`modeling_qwen3_omni_moe.py:678-693,818`).

### 3.3 Exact windowed attention

For batch item length `T_b`, total packed length is `A(T_b)`. Let `M=max_i A(c_i)` (normally 13), `ratio=floor(800/100)=8`, and `W=M*ratio` (normally 104). Partition each sample independently into consecutive, nonoverlapping windows `[W,W,...,remainder]`. `cu_seqlens` is the int32 cumulative sum beginning at zero over those window lengths (`modeling_qwen3_omni_moe.py:696-735`).

Thus “window 8 chunks” means each transformer attention call is fully bidirectional inside 104 packed positions corresponding to eight consecutive conv chunks, and has **zero attention across window boundaries**. It is not causal, not sliding/overlapping, and not a ±8-chunk neighborhood. The final tail is a shorter independent bidirectional sequence; it is neither padded into attention nor allowed to attend backward into the preceding full window. For batch >1, no window crosses a sample boundary.

The golden eager implementation materializes Q/K/V as `[1,20,S,64]`, splits dimension 2 using every adjacent `cu_seqlens` difference, runs each split independently, and concatenates outputs in order (`modeling_qwen3_omni_moe.py:526-590`). For each window:

```text
Q = Linear_bias(x,Wq,bq); K = Linear_bias(...); V = Linear_bias(...)
scores = matmul(Q, transpose(K)) * (1/sqrt(64))
P = softmax(scores, dim=-1, compute_dtype=float32).to(BF16)
O = matmul(P_BF16, V_BF16)
out = Linear_bias(concat_heads(O), Wo, bo)
```

There is no mask and no dropout in eval (`modeling_qwen3_omni_moe.py:475-497,517-524`). To match the oracle, use this eager math rather than flash attention or SDPA unless token equality is separately proven.

### 3.4 Each of 32 transformer layers

The block is pre-LayerNorm with ordinary affine LayerNorm and no explicit dropout calls:

```text
r = x
n = LayerNorm(x, gamma_attn, beta_attn, eps=1e-5)
a = window_attention(n)
x = r + a
r = x
n = LayerNorm(x, gamma_ffn, beta_ffn, eps=1e-5)
h = Linear(n, W_fc1, b_fc1)          # 1280→5120
h = GELU_exact_erf(h)
h = Linear(h, W_fc2, b_fc2)          # 5120→1280
x = r + h
```

(`modeling_qwen3_omni_moe.py:594-640`). There is no post-LN after either residual beyond the next pre-LN. PyTorch LayerNorm computes its reduction in the backend's promoted accumulation; the port must use F32 mean/variance and affine arithmetic, then cast at the same boundaries, subject to probes in §11.

After layer 31: affine `ln_post` (eps 1e-5), `proj1` with bias (1280→1280), exact erf GELU, and `proj2` with bias (1280→2048), producing packed `[A(T),2048]` BF16 (`modeling_qwen3_omni_moe.py:824-828`).

## 4. Adapter

For every packed audio vector independently:

```text
g = Linear(x, W_gate, bias=None)       # 2048→8192
a = SiLU(g) = g * sigmoid(g)
u = Linear(x, W_up, bias=None)         # 2048→8192
z = a * u
out = Linear(z, W_down, bias=None)     # 8192→2048
```

This exact order is `down(SiLU(gate(x)) * up(x))`; all linears are bias-free (`modeling_Moss.py:78-87`). Inputs, GEMM outputs, SiLU/multiply output, and final output are BF16 on the oracle path. Do not fuse gate/up in a way that changes BF16 GEMM selection/reduction without a token probe.

## 5. Prompt construction and audio embedding injection

### 5.1 Fixed template

The loader always supplies the vendored template (`loader.py:60-63`). With time markers disabled by default (`processing_Moss.py:37-43`), the exact prompt is:

```text
[151644, 872, 198, 151669]
+ [0] * A(T)
+ [151670, 151645, 198, 151644, 77091, 198]
```

This is `<|im_start|>user\n<|audio_start|>`, audio slots, then `<|audio_end|><|im_end|>\n<|im_start|>assistant\n` (`chat_template_default.py:25-46`). Iteration stops at the first `text_token`, so the template's final training EOS is not part of inference input (`processing_Moss.py:114-145`; `chat_template_default.py:42-52`). Prompt length is `A(T)+10`. The returned ordinary attention mask is all ones, but the golden decoder discards it and builds its own additive causal mask (`processing_Moss.py:201-212`; `reference.py:74-87`).

Concrete fixture prompts are therefore:

* short: `[151644,872,198,151669] + [0]×97 + [151670,151645,198,151644,77091,198]` (107 IDs);
* medium: same with `[0]×290` (300 IDs);
* long: same with `[0]×967` (977 IDs).

No separate prompt/meta files exist for these fixtures; these are fully extractable from fixture sample counts and the checked-in template.

### 5.2 `audio_input_mask` and scatter order

The mask has false for the 4-token prefix and 6-token suffix, true for every placeholder; token ID zero by itself is **not** the semantic test (`processing_Moss.py:125-145`). Build all normal token embeddings first, including embeddings for ID 0. Run the adapter over packed audio features. Expand mask `[1,L]→[1,L,2048]` as a stride-zero view and apply PyTorch `masked_scatter` (`reference.py:36-40`; model in-place equivalent `modeling_Moss.py:159-172`).

PyTorch consumes the flattened audio source in row-major order for each true element in flattened destination order. Therefore audio row 0 fills all 2048 channels of the first audio slot, row 1 the next slot, etc. Require `number_true_elements = A(T)*2048 = audio_embeds.numel()`. A simpler C++ row copy is equivalent only if it preserves this ordering exactly.

## 6. Qwen3 LLM decode contract

### 6.1 Shared per-layer math

Embedding lookup uses tied `[151936,2048]` BF16 weights; lm_head is the same storage, bias-free (`modeling_Moss.py:188-207`). For each of 28 full-attention layers:

1. Save `residual=x`.
2. **RMSNorm:** convert `x` to F32; `v=mean(x*x)` over 2048; `n_f32=x*rsqrt(v+1e-6)`; cast `n_f32` back to BF16; then multiply by the norm weight. This weight multiply occurs **after the cast**, exactly as `return self.weight * hidden_states.to(input_dtype)` (`modeling_qwen3.py:49-64`). Checkpoint norm weights are loaded BF16 in the PyTorch oracle even though a proposed GGUF may store them F32.
3. Bias-free Q/K/V projections. Shape projected Q as `[B,S,16,128]`, K/V as `[B,S,8,128]`.
4. Apply per-head Q/K RMSNorm over the last 128 elements **before transpose and before RoPE**: `q_proj → view(B,S,H,128) → q_norm → transpose(1,2)`; same for K; V has no norm (`modeling_qwen3.py:230-268`). Q/K norm uses the identical F32 reduction/cast-before-weight rule and eps 1e-6.
5. RoPE for explicit `position_ids`: `inv_freq[i]=1/(1e6^(2i/128)), i=0..63`. In disabled-autocast F32, compute matrix product of F32 inverse frequencies and F32 positions, duplicate frequencies by concatenation to 128, then `cos`/`sin`, then cast each to BF16 (`modeling_qwen3.py:124-148`). Broadcast over heads. `rotate_half([x0..x63,x64..x127])=[-x64..-x127,x0..x63]`; calculate `(q*cos) + (rotate_half(q)*sin)` and same for K in ATen BF16 operation order (`modeling_qwen3.py:151-180`). This is non-interleaved half rotation, not adjacent-pair rotation.
6. Append rotated K and unrotated V to the layer cache. Reference `StaticLayer` allocates `[1,8,max_cache_len,128]` in incoming BF16 dtype and uses `index_copy_` at an internal cumulative position (`transformers/cache_utils.py:364-445`). In this installed transformers revision, the explicit `cache_position` is required by mask plumbing but `StaticLayer.update` itself derives write indices from `cumulative_length`; C++ should simply write the explicit logical positions.
7. Repeat each KV head twice in order to match 16 Q heads (`modeling_qwen3.py:184-193`). Compute `scores=Q@K^T * 1/sqrt(128)`, add the supplied BF16 additive mask, softmax in F32, cast probabilities to BF16, then `P@V` in BF16 (`modeling_qwen3.py:196-218`). Dropout is zero.
8. Concatenate heads and bias-free O projection; `x = residual + attn_out` (`modeling_qwen3.py:289-290,315-327`).
9. Save residual; post-attention RMSNorm with the same exact rule; bias-free `gate`, `up`; `SiLU(gate)*up`; bias-free `down`; residual add (`modeling_qwen3.py:70-83,329-334`).

After layer 27 apply final RMSNorm. Compute tied lm_head only for the last sequence row and choose `argmax` over vocab. PyTorch argmax selects the lowest index on ties.

### 6.2 Exact prefill and decode loop

Golden settings are batch 1, `max_new_tokens=200`, EOS 151645, and **max cache length 2048** because the capture script overrides the helper default (`scripts/moss_golden.py:57-64`; `reference.py:49-68`). It asserts `T+200<=2048`.

Let BF16 minimum finite value `NEG=-3.3895313892515355e38` (`torch.finfo(torch.bfloat16).min`). Prefill:

```text
position_ids = [[0,1,...,T-1]]              # int64
cache_position = [0,1,...,T-1]              # int64
mask[0,0,i,j] = BF16(0.0) if j<=i else NEG  # shape [1,1,T,2048]
out = Qwen3Model(inputs_embeds=merged_prompt, mask, position_ids,
                 StaticCache, use_cache=true, cache_position)
first = argmax(tied_head(out.last_hidden_state[:,T-1:T,:]))
```

(`reference.py:70-90`). A 4-D mask is mandatory: it makes `create_causal_mask` accept the already prepared mask. A `[1,1,1,N]` mask during prefill is wrong because it omits per-query causality (hard-won warning in `comms.md:675-676`). Explicit `cache_position` must also be supplied.

Then for `i=0..198`, with `cur=T+i`:

```text
input_ids = [[previous_generated_token]]
position_ids = [[cur]]
cache_position = [cur]
mask[0,0,0,j] = 0 if j<=cur else NEG         # [1,1,1,2048]
out = model(...)
next = argmax(tied_head(last row))
append next
stop iff next == 151645
```

(`reference.py:92-112`). Notice EOS is checked only after decode-loop tokens; the prefill-produced first token is appended without an immediate EOS check. This quirk must be copied. If no EOS arrives, exactly 200 IDs are returned; the long golden demonstrates this. Generated IDs exclude prompt IDs.

Existing golden lengths are short 31 (ends EOS), medium 89 (ends EOS), and long 200 (truncated without EOS). Their complete IDs are stored in the `.pt` files; those files, not a transcription printed in this document, are the acceptance oracle (`scripts/moss_golden.py:64-69`).

## 7. Detokenization

The checkpoint requests `Qwen2Tokenizer`, a GPT-2-style byte-level BPE tokenizer (`HF tokenizer_config.json:229-238`). Source assets are `vocab.json`, `merges.txt`, `tokenizer.json`, `added_tokens.json`, `special_tokens_map.json`, and `tokenizer_config.json` in the HF snapshot (loader at `src/starling/moss/loader.py:60`).

For exact generated-id decoding, implement the standard Qwen2/GPT-2 decoder:

1. Map each non-special ID to its vocabulary token string.
2. Concatenate token strings with no separator.
3. Apply the reversible GPT-2 byte-decoder map (Unicode surrogate alphabet back to bytes).
4. Decode bytes as UTF-8 with `errors="replace"`.
5. `skip_special_tokens=True`: omit IDs marked special by tokenizer metadata, including 151645. Do not trim, lowercase, normalize spaces, or run generic “cleanup”; `clean_up_tokenization_spaces=false`, `add_prefix_space=false`, `errors="replace"` (`HF tokenizer_config.json:229-238`).

Using GGUF `tokenizer.ggml.model="gpt2"`, `tokenizer.ggml.tokens`, token types, merges, BOS/EOS/pad IDs is acceptable if converter output is byte-for-byte equivalent. Preserve all 151,936 token entries, including added tokens and their special/non-special classification. Golden text is written directly from `tokenizer.decode(ids[0], skip_special_tokens=True)` with no newline added (`scripts/moss_golden.py:64-69`).

## 8. Numerical risk register and required ggml cast policy

The acceptance test is generated IDs/text, but cross-engine differences compound. Start with the strict policy below.

| Point | Severity | Divergence mechanism | Required mitigation |
|---|---|---|---|
| FFT/mel frontend | critical | NumPy uses f64 FFT inputs, complex64 storage, f64 magnitude/dot behavior, then f32 | Prefer baked F32 640-window/321×128 bank; reproduce operation sequence; add mel fixture goldens before optimization. |
| n_fft 400 vs 640 | critical | Different features entirely | Use 640 for Starling oracle; reject cstr's 400 constants. |
| mel global clamp | critical | Per-frame clamp or ln changes all features | Base-10 log, global max−8, then `(x+4)/4`. |
| BF16 mel cast | high | Encoder sees quantized mel | Cast F32 mel to BF16 before conv. |
| conv padding/chunk boundaries | critical | receptive field and token count change | Three 3×3 stride-2 pad-1 convs independently on each 100-frame piece. |
| conv/GEMM precision | critical | ggml F16 file or F32 activation changes reductions | Exact profile uses BF16 weights/activations and backend BF16 GEMMs; compare stage goldens. |
| GELU variant | critical | tanh approximation compounds through 35 uses | exact erf GELU only. |
| sinusoid construction | high | f32 exp/sin/cos differs from host doubles | Generate with the recorded Torch F32 formula or store the 1500×1280 F32 table in GGUF. |
| ordinary LayerNorm | critical | eps/variance/affine dtype | F32 reduction, eps 1e-5, affine params; cast at PyTorch-equivalent boundary. Probe exact affine order. |
| window semantics | critical | global/sliding/causal attention is another model | nonoverlapping bidirectional windows of normally 104 packed tokens; isolated tail. |
| attention softmax | critical | BF16 softmax loses rank information | scores/mask as reference, F32 softmax, cast probabilities to BF16 before `P@V`. |
| adapter scatter | critical | feature rows shifted into prompt | row-major slot copy exactly matching flattened masked_scatter. |
| RMSNorm | critical | ggml `rms_norm` commonly accepts only F32 and may multiply weight before cast | `BF16 x → F32 normalize → BF16 normalized → BF16 weight multiply`; do not keep normalized activation F32 through the weight multiplication. |
| Q/K norm order | critical | changes every attention score | projection→view `[B,S,H,D]`→per-head norm→transpose→RoPE. |
| RoPE | critical | fused kernels round BF16 products differently; wrong rotate convention | ATen-style non-interleaved rotate-half; F32 frequency/cos/sin generation then BF16; separate mul, mul, add. Fused Triton RoPE was rejected for borderline tokens (`comms.md:674`). |
| RoPE positions | critical | off-by-one ruins decode | explicit 0..T−1 prefill, then T+i. |
| KV cache | critical | unrotated K, wrong layout, or wrong write slot | cache rotated K and raw V as BF16 `[layer,1,8,2048,128]`; explicit logical position. |
| 4-D causal mask | critical | prefill can see future prompt/audio slots | exact `[1,1,T,2048]`; finite BF16 minimum, not an arbitrary −1e9. |
| GQA repetition | high | head ordering changes | KV head order `[0,0,1,1,...,7,7]`. |
| residual/SiLU dtype | high | F32 stream diverges over 28 layers | cast/store BF16 after each PyTorch BF16 elementwise result; preserve `SiLU(gate)*up` order. |
| tied head/argmax | critical | F16 embedding copy changes logits | exact same embedding tensor for input and head; lowest-ID tie behavior. |
| F16 conversion | critical | checkpoint is BF16, F16 is not value-preserving | Byte-exact profile must store BF16. F16 is a separate approximate deployment profile and cannot claim oracle equivalence. |
| long tiled BF16 GEMMs | critical | cuBLAS/ggml reduction and algorithm choices flip borderline argmax | deterministic algorithm selection; long golden every run. Existing Python notes report run-to-run long flips (`comms.md:675`). |
| fused/compiled substitutions | high | mathematically equivalent is not bit equivalent | Gate every substitution on full short/medium/long ID equality. |

### ggml elementwise constraint

Starling's prior ggml study found CUDA `rms_norm`, multiply, add, SiLU, and scale paths effectively F32-only/assert F32, creating an F32 activation stream and eventual token divergence (`comms.md:681`). Therefore graphs should explicitly encode storage and arithmetic boundaries:

* **Linear/conv input:** BF16 storage. Matmul/conv output BF16 for exact profile.
* **RMSNorm:** BF16→F32; ggml F32 rms norm; **cast F32 normalized value to BF16**; cast norm weight to BF16 if stored F32; BF16 multiply (a custom kernel may be required), output BF16.
* **LayerNorm:** BF16→F32 reduction and affine; probe whether PyTorch's affine rounds only at final BF16 store; implement a custom kernel if generic ggml differs.
* **GELU/SiLU/residual/multiply:** if generic ops require F32, cast BF16 inputs to F32, execute the single reference operation, and cast immediately back to BF16 at the point ATen emits a BF16 tensor. Do not let F32 survive into the next GEMM.
* **Attention:** BF16 Q/K matmul and scale/mask matching ATen; F32 softmax; BF16 probability store; BF16 P×V.
* **RoPE:** F32 cos/sin generation → BF16 cos/sin; BF16 products and BF16 add. A custom CUDA op is safer than generic all-F32 glue.

These casts reproduce *dtype semantics*, not necessarily cuBLAS reduction order. Byte-exact token acceptance remains the final gate.

## 9. Starling GGUF proposal

### 9.1 Reference file findings and deliberate deviations

`import gguf` resolves from the repo venv. The cstr F16 GGUF is v3, architecture `moss_transcribe`, 840 tensors. Its useful naming/layout is compact and should be retained. It stores matrix weights F16, biases/norms F32, `llm.embed.weight` F16, and no separate lm_head; it includes GPT-2 tokenizer metadata. It also stores `audio.mel_filters [128,201]` and `audio.mel_window [400]`—the incorrect frontend for Starling's oracle.

Proposed profiles:

* `moss-bf16-exact` (**required for this spec**): all checkpoint learned tensors BF16, including embeddings and norm weights, except frontend constants and optional generated sinusoid table F32. Biases originate BF16 in the loaded oracle and should be BF16 too. This maximizes the chance of matching PyTorch BF16.
* `moss-f16` (shipping/compatibility, not byte-exact): follow cstr policy—matrix/embedding F16 and biases/norms F32. It may be useful, but it must have separate goldens and must not be advertised as matching the PyTorch BF16 IDs.

### 9.2 Tensor name map

HF matrices are logically `[out,in]`; GGUF/ggml shape display is often reversed. Converter tests must compare logical dimensions, not printed tuple order.

| HF-native name | Starling GGUF name |
|---|---|
| `model.audio_model.conv2d{1,2,3}.{weight,bias}` | `enc.conv{1,2,3}.{weight,bias}` |
| `model.audio_model.conv_out.weight` | `enc.conv_out.weight` |
| generated/stored sinusoid table | `enc.positional_embedding` (optional F32; preferred for numerical stability) |
| `model.audio_model.layers.{i}.self_attn_layer_norm.{weight,bias}` | `enc.blk.{i}.attn_norm.{weight,bias}` |
| `...layers.{i}.self_attn.{q_proj,k_proj,v_proj,out_proj}.{weight,bias}` | `enc.blk.{i}.attn.{q,k,v,o}.{weight,bias}` |
| `...layers.{i}.final_layer_norm.{weight,bias}` | `enc.blk.{i}.ffn_norm.{weight,bias}` |
| `...layers.{i}.fc1.{weight,bias}` | `enc.blk.{i}.ffn.fc1.{weight,bias}` |
| `...layers.{i}.fc2.{weight,bias}` | `enc.blk.{i}.ffn.fc2.{weight,bias}` |
| `model.audio_model.ln_post.{weight,bias}` | `enc.ln_post.{weight,bias}` |
| `model.audio_model.proj{1,2}.{weight,bias}` | `enc.proj{1,2}.{weight,bias}` |
| `model.audio_adapter.{gate_proj,up_proj,down_proj}.weight` | `adapter.{gate,up,down}.weight` |
| `model.language_model.embed_tokens.weight` | `llm.embed.weight` |
| `model.language_model.layers.{i}.input_layernorm.weight` | `llm.blk.{i}.attn_norm.weight` |
| `...self_attn.{q_proj,k_proj,v_proj,o_proj}.weight` | `llm.blk.{i}.attn.{q,k,v,o}.weight` |
| `...self_attn.{q_norm,k_norm}.weight` | `llm.blk.{i}.attn.{q_norm,k_norm}.weight` |
| `...post_attention_layernorm.weight` | `llm.blk.{i}.ffn_norm.weight` |
| `...mlp.{gate_proj,up_proj,down_proj}.weight` | `llm.blk.{i}.ffn.{gate,up,down}.weight` |
| `model.language_model.norm.weight` | `llm.final_norm.weight` |
| `lm_head.weight` | omitted; alias `llm.embed.weight` |
| Whisper Slaney bank/window | `audio.mel_filters` `[128,321]`, `audio.mel_window` `[640]`, F32 |

### 9.3 Required metadata

Retain cstr's `general.architecture=moss_transcribe` and add/profile-version keys:

```text
starling.format_version = 1
starling.numeric_profile = "bf16_exact" | "f16"
moss_transcribe.enc.{num_mel_bins=128,encoder_layers=32,d_model=1280,
  encoder_attention_heads=20,head_dim=64,encoder_ffn_dim=5120,
  downsample_hidden_size=480,max_source_positions=1500,n_window=50,
  n_window_infer=800,conv_chunksize=500,output_dim=2048,layer_norm_eps=1e-5}
moss_transcribe.adapter.{input_size=2048,hidden_size=8192,output_size=2048}
moss_transcribe.llm.{hidden_size=2048,num_layers=28,num_heads=16,
  num_kv_heads=8,head_dim=128,intermediate_size=6144,vocab_size=151936,
  max_position_embeddings=40960,rope_theta=1000000,rope_scaling="none",
  rms_norm_eps=1e-6,tied_embeddings=true}
moss_transcribe.frontend.{sample_rate=16000,n_fft=640,win_length=640,
  hop_length=160,n_mels=128,center=true,pad_mode="reflect",power=2,
  mel_scale="slaney",mel_norm="slaney",log="log10",mel_floor=1e-10,
  dynamic_range=8,normalization_offset=4,normalization_divisor=4,
  output_dtype="bf16"}
moss_transcribe.{pad_token_id=151643,eos_token_id=151645,start_token_id=151644,
  audio_start_id=151669,audio_end_id=151670,audio_placeholder_id=0,
  max_new_tokens=200,max_cache_len=2048}
```

Store the prompt prefix/suffix as integer arrays to prevent template drift. Store standard GGUF GPT-2 tokenizer tokens, scores/types, merges, special-token IDs, and special-token classification. cstr's metadata is missing several frontend and runtime contracts above; Starling must add them.

## 10. C++ module sketch

Follow the small value-object config and loader ownership style in `cpp/parakeet/config.hpp:12-58` and `cpp/parakeet/loader.hpp:13-29`, and the exception-fenced flat ABI in `cpp/include/starling_ggml.h:1-86`.

```text
cpp/moss/
  config.hpp/.cpp          metadata structs, validation, derived lengths
  loader.hpp/.cpp          MossModel { Config; ModelLoader; tensor handles }
  mel.hpp/.cpp             exact CPU NumPy-compatible Whisper frontend
  audio_encoder.hpp/.cpp   chunk/conv/packed window encoder graph
  adapter.hpp/.cpp         gated MLP
  prompt.hpp/.cpp          IDs, mask, embedding row injection
  llm.hpp/.cpp             prefill/decode graphs, KV cache, greedy loop
  tokenizer.hpp/.cpp       Qwen2 byte-BPE detokenizer
  capi_moss.cpp            model dispatch and C ABI implementation
```

Recommended public C++ surface:

```cpp
namespace starling::ggml::moss {
struct FrontendConfig { uint32_t sample_rate, n_fft, win_length, hop_length, n_mels; /* log fields */ };
struct EncoderConfig { uint32_t n_layers, d_model, n_heads, head_dim, ff_dim; /* conv/window fields */ };
struct LlmConfig { uint32_t n_layers, hidden, n_heads, n_kv_heads, head_dim, intermediate, vocab, max_cache; /* rope */ };
struct Config { FrontendConfig frontend; EncoderConfig encoder; LlmConfig llm; /* adapter/tokens/tokenizer */ };

int64_t audio_token_length(int64_t mel_frames);

struct MossModel {
    Config config;
    ModelLoader loader;
    bool load(const char* gguf_path, std::string& err);
};

struct MelFeatures { std::vector<ggml_bf16_t> data; int64_t n_mels, n_frames; };
bool compute_log_mel(const Config&, std::span<const float> pcm16k, MelFeatures&, std::string& err);

struct AudioEncoding { /* device tensor/owned graph result */ int64_t n_tokens; };
bool encode_audio(MossModel&, const MelFeatures&, AudioEncoding&, std::string& err);
bool apply_adapter(MossModel&, const AudioEncoding&, /*out device tensor*/, std::string& err);

struct Prompt { std::vector<int32_t> ids; std::vector<uint8_t> audio_mask; };
Prompt build_transcribe_prompt(const Config&, int64_t mel_frames);
bool build_inputs_embeds(MossModel&, const Prompt&, /*adapter output*/, /*out*/, std::string& err);

struct GenerateOptions { int32_t max_new_tokens = 200; int32_t max_cache_len = 2048; int32_t eos_token_id = 151645; };
struct GenerateResult { std::vector<int32_t> ids; bool hit_eos = false; };
bool greedy_generate(MossModel&, /*prompt embeds*/, const GenerateOptions&, GenerateResult&, std::string& err);

class Tokenizer {
public:
    bool load(const gguf_context*, const Config&, std::string& err);
    std::string decode(std::span<const int32_t> ids, bool skip_special_tokens=true) const;
};

bool transcribe_pcm(MossModel&, std::span<const float> pcm16k, GenerateResult&, std::string& text, std::string& err);
} // namespace starling::ggml::moss
```

Keep the existing external API `starling_ggml_load(STARLING_GGML_MOSS, ...)` and `starling_ggml_transcribe_pcm(...)` (`cpp/include/starling_ggml.h:34-80`). `capi_moss.cpp` should be model-specific internally, while shared dispatch remains exception-fenced and returns malloc-owned UTF-8 exactly as the header specifies.

Implementation should expose stage-level test hooks only in C++ tests (mel, encoder output, adapter output, merged embeddings, prefill logits, first decode step); do not enlarge the stable C ABI for probes unless needed by Python integration.

## 11. Acceptance sequence and open empirical questions

### Mandatory bring-up order

1. Validate GGUF tensor names, logical dimensions, aliases, and profile dtype.
2. Capture/check CPU mel tensors for all three fixtures before model work.
3. Compare conv output, packed indices, `cu_seqlens`, each selected encoder layer, ln_post/projection, adapter, and merged prompt embeddings against new CPU-saved probes.
4. Compare prefill last hidden/logits and one decode step with cache slices.
5. Require complete generated ID equality against all three `.pt` files and exact text-file byte equality.
6. Run long repeatedly to detect nondeterministic borderline flips.

### Open questions requiring probe scripts

These are deliberately not guessed in the contract:

- [ ] **Frontend cross-language exactness:** can the intended C++ FFT/mel implementation reproduce the NumPy f64→complex64→f64 path closely enough, or must Starling preserve a compatibility CPU kernel/reference FFT?
- [ ] **Effective n_fft fixture proof:** save processor mel shape/filter shape and first/last mel columns to permanently demonstrate the loader's 640 behavior. Source inspection resolves it, but a regression artifact is valuable.
- [ ] **LayerNorm exact cast boundary:** verify ordinary BF16 `nn.LayerNorm` output against candidate F32-reduce/affine kernels, especially whether weight/bias arithmetic rounds only once.
- [ ] **BF16 norm-weight ordering in ggml:** verify custom RMSNorm implements F32 normalize → BF16 cast → BF16 weight multiply, not F32 weight multiply.
- [ ] **Conv and GEMM algorithm sensitivity:** determine whether ggml CUDA BF16 kernels preserve the fixture argmax path; if not, identify required cuBLAS algorithm/workspace settings.
- [ ] **Attention score scaling order:** probe whether ATen's `matmul(...) * scalar` rounds the matmul to BF16 before scaling on this build, and mirror it explicitly.
- [ ] **Elementwise BF16 boundaries:** save outputs around GELU, SiLU×up, RoPE products/add, and residual adds to validate custom kernels.
- [ ] **Sinusoid generation:** compare generated C++ table to Torch's F32 table; if not exact enough, mandate stored table tensor.
- [ ] **Tokenizer GGUF completeness:** round-trip every vocab ID and all golden IDs against `Qwen2Tokenizer.decode`, including invalid UTF-8 and non-special added tokens.
- [ ] **F16 deployment profile:** establish its own expected IDs; it cannot inherit the BF16 exact claim.
- [ ] **Maximum supported audio:** choose a public limit from `prompt_len + max_new_tokens <= max_cache_len`; current fixed 2048 cache supports the 74-second fixture but not arbitrarily long audio.
- [ ] **Long-run stability:** characterize and eliminate the reported cuBLAS BF16 long-decode nondeterminism before claiming repeatable byte exactness across hardware/driver versions.

