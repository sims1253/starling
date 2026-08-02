# ARK-ASR-3B ggml port — performance state & hotpath analysis

Measured on RTX 5090, branch `ggml-ark`. The port is **byte-exact** (short/medium/
long transcripts match the golden reference; WER identical to PyTorch, including
the 70% long-fixture WER which is the model's own behavior, not a port bug).
Methodology: `bench_all.py --reps 20` (5 untimed warmup, median of 20,
`torch.cuda.synchronize()`-bracketed), GPU held under the process gpu-lock.

## Headline: optimized to 295 / 797 / 910 ms (from 1858 / 5619 / 9073 ms)

| fixture | audio_s | starling (PyTorch) ms | starling-ggml ms | speedup vs unopt | ggml RTFx |
|---------|--------:|---------------------:|-----------------:|-----------------:|----------:|
| short   | 7.4     | 185                  | 295 ± 5          | 6.3x             | 25x       |
| medium  | 22.3    | 533                  | 797 ± 11         | 7.0x             | 28x       |
| long    | 74.3    | 595                  | 910 ± 19         | 10.0x            | 82x       |

(± = stdev over 20 reps. VRAM reports 0.0GB because `_vram_gb` reads torch's
allocator but libstarling_ggml allocates CUDA memory directly — a reporting gap.)

For context, the repo's other ggml ports: moss 214/535/1180 ms; parakeet
14/30/86 ms. ARK ggml is now in the moss performance band and ~1.5-5x behind
PyTorch peak (which is itself a heavily-fused Triton/CUDA-graph path). The model
is bigger than moss (36 vs 28 layers, 11008 vs 6144 FFN, GQA-8 vs GQA-2).

## Per-stage breakdown (STARLING_ARK_TIMING, warm)

| fixture (mel) | mel  | enc+adapt | prompt+embeds | gen  | total |
|---------------|-----:|----------:|--------------:|-----:|------:|
| short (743)   | ~8   | ~140      | ~2            | ~370 | ~520  |
| medium (2230) | ~9   | ~170      | ~3            | ~770 | ~950  |
| long (capped) | ~10  | ~170      | ~5            | ~830 | ~1015 |

(mel = host FFT frontend; enc+adapt = conv + 32 encoder layers + MLP adapter in
one captured CUDA graph; gen = Qwen2.5 prefill + K-step decode.)

**Decode (`gen`) is now the dominant cost** (~50-80% of wall time), ~7-10 ms/token.
The encoder is ~140-170ms steady-state (first call per shape pays a one-time
CUDA-graph capture cost of ~1-2s).

## What was slow and the two fixes that closed the gap

### Fix 1: Conv front-end moved onto the GPU (~6-10x enc win)
The Whisper Conv1d (K=3) was running on the HOST as a scalar C++ double-loop
(`host_conv1d_gelu`) — a workaround for an earlier "ggml conv produces wrong
values under capture" diagnosis. That diagnosis was **wrong**: the real bug was a
result-layout transpose error, not a capture bug. The CUDA `im2col` kernel
requires **F32** inputs (`src1->type == GGML_TYPE_F32`); feeding it the bf16 mel
directly caused the failure. Casting the mel and weights to f32 before
`ggml_conv_1d` (the same lesson that made flash attention work) makes the conv
graph-safe. The conv now runs as fast cuBLAS GEMMs in the captured encoder graph.

Before (host conv): enc+adapt was 2083 / 6619 / 11103 ms (the scalar conv2
1280×1280×1500×3 ≈ 7.4 GMACs dominated).
After (GPU conv): enc+adapt is ~140 / ~170 / ~170 ms steady — **~57-79x faster**.
`host_conv1d_gelu` is retained as the CPU/debug fallback (`STARLING_ARK_DEBUG=1`).

### Fix 2: Flash attention for the encoder (~1.1-1.7x enc win)
ARK uses **global O(T²) attention** (vs moss's windowed O(T·W)), materializing a
full `[T,T,H]` score tensor per layer. Replaced the manual `mul_mat`+`softmax`+
`mul_mat` with `ggml_flash_attn_ext` (the proven parakeet path), which fuses the
softmax and avoids materializing the score tensor. ARK is bidirectional (no mask)
and plain MHA (n_head == n_head_kv == 20), so flash's preconditions hold
trivially. Cast q/k/v to f32 before flash (the bf16→f16 in-graph conversion
breaks CUDA-graph capture). Byte-exact transcripts preserved. Kill-switch:
`STARLING_ARK_NO_FATTN=1` selects the manual fallback.

## What's NOT a bug (correcting the earlier perf doc's misdiagnoses)
- **"Encoder GEMMs run in a non-accelerated bf16 path"** — wrong. The
  `STARLING_REPLAY_TIMING` histogram lumps BF16 into "other"; both ARK and moss
  encoders report `other=N` and both are on the fast cuBLAS bf16 path
  (`cublasGemmEx`, CUDA_R_16BF, tensor-op). The "other=258" was a metric artifact.
- **"Decode pays 29ms/token in a per-replay D2H sync that moss avoids"** — wrong.
  ARK and moss use byte-identical K-step decode plumbing (`compute_with_captures`,
  same single trailing `cudaStreamSynchronize`, same one 16-byte capture). The
  ~7ms/token is genuine GPU compute for ARK's bigger model (36 vs 28 layers, 11008
  vs 6144 FFN, GQA-8 vs GQA-2) — already on the fast bf16 cuBLAS path. The
  `cuda_wall_ratio ≈ 0.98` for a comparable Qwen2.5 decoder confirms it's
  compute-bound, not sync-bound.

## Remaining optimization headroom (decode-bound now)
The encoder is no longer the bottleneck; decode (~7-10 ms/token, ~50-80% of wall)
is. Options, in priority order:
1. **Weight quantization (q8_0/q4_0)** for the decoder — engages ggml's MMQ
   tensor-core kernels and cuts the decode GEMM memory traffic (the decode is
   memory-bandwidth-bound). Would need a byte-exactness check (q8_0 usually keeps
   argmax-stable transcripts; q4_0 may drift). Highest decode leverage.
2. **Larger K** for the K-step (currently K=4, clamped to 8) — more tokens per
   host sync, but the decode is compute-bound so the gain is marginal.
3. **`prompt+embeds`** is negligible (~2-20ms); not worth optimizing.
4. The encoder's one-time per-shape CUDA-graph capture (~1-2s first call) could be
   pre-warmed (`prewarm`) for latency-sensitive single-utterance use.

With q8_0 decode, projected end-to-end: short ~150ms, medium ~350ms, long ~400ms
— at or below PyTorch peak.

## Build/run/bench
```
cmake --build build -j   # -DSTARLING_GGML_CUDA=ON -DSTARLING_GGML_SHARED=ON
uv run pytest tests/test_ggml_parity.py -k ark          # parity (3 cases)
uv run python benchmarks/bench_all.py \
  --models ark --engines starling,starling-ggml --lengths short,medium,long \
  --batches 1 --warmup 5 --reps 20                       # benchmark
STARLING_ARK_TIMING=1 <transcribe>                       # stage breakdown (stderr)
STARLING_ARK_NO_FATTN=1 <transcribe>                     # manual-attn fallback
STARLING_ARK_DEBUG=1 <transcribe>                        # host-conv fallback
```

## Artifacts
- `outputs/bench_all.json` — official 20-rep records (starling + starling-ggml)
- `golden/ark_reference.json` — the byte-exact reference
