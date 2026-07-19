# In-tree ggml Parakeet baseline and gap analysis

## Scope and protocol

This is a steady-state, B=1 comparison of Starling's **in-tree** `libstarling_ggml.so` engine (`StarlingGgmlParakeet`) and the PyTorch peak engine (`ParakeetStarling`, engine key `starling`). It is not a result for the separate external `parakeet.cpp` server discussed in [ggml-parakeet-perf-analysis.md](ggml-parakeet-perf-analysis.md).

Hardware was an NVIDIA GeForce RTX 5090 (Torch 2.12.1+cu130). Before each timed session, `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` reported no compute applications. Timed runs used `with_gpu_lock(session="ggml-baseline", model="parakeet-tdt-0.6b-v3")`.

For each engine, the model was loaded once, then every fixture received five untimed `_run_one(audio)` calls followed by 20 wall-clock repetitions. CUDA was synchronized immediately before and after each sample so the PyTorch GPU work was included; the sample is `time.perf_counter()` around `_run_one`. Audio durations are 7.435 s / 22.305 s / 74.350 s. Both engines produced the existing byte-exact transcript (harness WER 0.00% for every fixture).

## Baseline

`±` is sample standard deviation, not a confidence interval. RTFx is audio duration divided by median latency. “Gap” is in-tree ggml median / PyTorch median; lower is better.

| fixture | audio | in-tree ggml median ± stdev (min) | PyTorch peak median ± stdev (min) | ggml / peak gap | ggml RTFx | peak RTFx |
|---|---:|---:|---:|---:|---:|---:|
| short | 7.435 s | **149.9 ± 31.8 ms** (81.0 ms) | **16.4 ± 1.9 ms** (13.8 ms) | **9.16×** | 49.6× | 454.2× |
| medium | 22.305 s | **243.6 ± 43.8 ms** (187.7 ms) | **25.2 ± 1.3 ms** (24.3 ms) | **9.66×** | 91.6× | 884.8× |
| long | 74.350 s | **438.8 ± 97.7 ms** (298.1 ms) | **59.4 ± 1.4 ms** (58.2 ms) | **7.38×** | 169.4× | 1,251.2× |

The in-tree engine has materially higher variance than the PyTorch path, including a wide short-fixture range. Therefore the median is the baseline; the best individual ggml samples are not treated as its steady-state result.

### Harness cross-check

`benchmarks/bench_all.py` can select both engines directly:

```bash
uv run python benchmarks/bench_all.py \
  --models parakeet --engines starling,starling-ggml \
  --lengths short,medium,long --batches 1 --warmup 5 --reps 20
```

It was run without `--update-readme`, under its own GPU-lock session. Its medians were 141.2 / 253.0 / 388.2 ms for in-tree ggml and 16.9 / 27.8 / 61.5 ms for PyTorch (short / medium / long), equivalent to 8.35× / 9.10× / 6.31× gaps. That is directionally consistent with the direct protocol, while the ggml spread explains the run-to-run median movement. The direct table above remains the requested `_run_one` baseline.

## Phase timing availability

No supported phase-timing gate exists in the in-tree engine, so no mel/encoder/decode breakdown was collected and no instrumentation was added. A source search for `getenv`, `PARAKEET_*TIMING`, and `ENCTIMING` found only mel debug/dump controls: `STARLING_MEL_DEBUG` in `cpp/parakeet/capi_parakeet.cpp:62` and `cpp/parakeet/mel.cpp:31`, plus `STARLING_MEL_DUMP_CPU`, `STARLING_MEL_DUMP`, and `STARLING_MEL_DUMP_XW` in `cpp/parakeet/mel.cpp:176,362,373`. These are diagnostic dumps, not phase clocks. The device selector is `STARLING_GGML_DEVICE` (`cpp/runtime/backend.cpp:83`).

## Code-grounded gap analysis

The external-engine analysis identifies device-resident decode state, speculative/K-step decode, and an on-device argmax megakernel as the route to parity. Those are still open in the in-tree implementation; importantly, this source tree is at an earlier performance stage than that external engine.

### Already implemented

* **Mel graph caching is present.** GPU mel owns a per-length `ReplayGraph`: `GpuMel::build_or_reuse_replay` returns on a matching frame count (`cpp/parakeet/mel.cpp:242-244`) and constructs the graph once (`:253`). The persistent frontend is created once at model load (`cpp/parakeet/capi_parakeet.cpp:68-69`). This rules out “add mel graph caching” as the first medium/long optimization. Host-side preemphasis/framing/windowing and CMVN remain (`cpp/parakeet/mel.cpp:306-326` and `:397+`) to preserve the current exact numeric path.
* The backend contains a general `ReplayGraph` implementation and documents persistent allocation / async transfer / single synchronization (`cpp/runtime/backend.cpp:1-11`). That infrastructure is available, but the Parakeet encoder and decoder below do not use it.

### Open and prioritized work

1. **Make greedy decode device-resident and capture a multistep decode graph.** This is the largest and most direct application of the external analysis. The current decoder is a host-controlled nested serial loop (`cpp/parakeet/tdt.cpp:44-49`), calls `PredictionNet::step` (`:57`) and `Joint::step_argmax` (`:63`) for each step, and keeps committed LSTM state as host `std::vector<float>` (`:31`, `cpp/parakeet/prediction.cpp:24-28`). Each prediction step builds and executes a graph (`cpp/parakeet/prediction.cpp:71-111`) and captures every layer's h/c output into host vectors (`:105-106`). There is no K-step/multistep decoder implementation in `cpp/parakeet/tdt.cpp`, `prediction.cpp`, or `joint.cpp`. Keep h/c, last token, frame, and termination state in persistent device buffers, capture fixed-K transitions, and synchronize/read back only what is required at a chunk boundary. This addresses both the external document's device-state option and its K-step option.

2. **Fuse joint logits, argmax, duration choice, and loop control on device.** `Joint::step_logits` currently builds a graph per step (`cpp/parakeet/joint.cpp:51-75`), returns a host `std::vector<float>`, then `step_argmax` scans token and duration logits on the CPU (`:85-102`). The source explicitly labels it the CPU/host-argmax path (`:80-81`). A device argmax plus blank/duration/frame transition in the captured decode graph removes the full-vocabulary logits readback and is the in-tree equivalent of the external analysis's on-device argmax megakernel.

3. **Cache/capture the encoder by shape, then enable the missing GPU attention fast path.** `Encoder::encode` calls one-shot `run_graph` (`cpp/parakeet/encoder.cpp:32`) and its own comment says the cached-per-shape `ReplayGraph` version is future work (`:9-10`). It also reconstructs the 24-layer stack per invocation (`:49-51`). More critically, relative-position attention explicitly does **not** emit its GPU flash-attention path (`cpp/parakeet/relpos_attention.cpp:8-9`). Capturing the encoder and implementing the GPU flash-attention route are open; unlike mel, neither is already cached in this path. This can reduce encoder overhead but should follow decode because decode is repeated per emitted/frame step while encoder runs once per clip.

4. **Persist decode objects and inputs across calls; eliminate repeated graph construction and transfers before changing algorithmic semantics.** `parakeet_full_decode` creates `Encoder`, `PredictionNet`, and `Joint` for every transcription (`cpp/parakeet/capi_parakeet.cpp:179,202-203`). The prediction net also performs host input construction and registers h/c graph inputs every step (`cpp/parakeet/prediction.cpp:60-89`); the joint similarly registers both encoder and prediction host inputs (`cpp/parakeet/joint.cpp:57-63`). Move these to per-model, shape-keyed persistent decode graphs/buffers. This is lower-risk groundwork for items 1–2 and preserves the present exact control flow before introducing speculative rollback.

5. **Only then investigate speculative lookahead/rollback.** The external report's speculative option remains absent: the present loop immediately commits/discards state based on each `k` (`cpp/parakeet/tdt.cpp:69-76`) and advances frames based on that duration (`:79-80`). A rollback-capable fixed lookahead may reduce serial boundaries further, but it is higher risk to byte exactness than device-resident K-step replay and on-device argmax.

## Conclusion

In-tree ggml is not near the external engine's reported 16/38/108 ms reference: its direct steady-state medians are 150/244/439 ms and its medium/long deficit versus the PyTorch peak engine is 9.66×/7.38×. The immediate gap is not an unimplemented mel cache—one exists—but uncaptured, host-mediated serial decode plus an encoder that still uses one-shot graph construction and lacks the GPU flash-attention route. Prioritize persistent device-state K-step decode and device argmax, then encoder capture/attention fusion.


## Wave-3 encoder verification

Wave 3 added a GPU-only, per-mel-shape encoder `ReplayGraph` cache, the proven
per-head relative-position flash-attention path, and persistent model-bound
encoder / prediction / joint objects. `STARLING_PARAKEET_TIMING=1` now prints
host-wall mel, encoder, decode, and total phases. The CPU path remains the
original one-shot manual-attention reference; `STARLING_PARAKEET_NO_FATTN=1`
selects that manual attention graph on GPU.

The CUDA measurement used the same RTX 5090 and fixtures as the original
baseline. `nvidia-smi` showed no compute applications before measurement. The
model was loaded once under `with_gpu_lock(session="ggml-w3v",
model="parakeet-tdt-0.6b-v3")`; for each fixture, five `_run_one(audio)` calls
were discarded and the following 20 calls were measured with
`time.perf_counter()`. The phase columns are medians of the corresponding native
host-wall timing lines over those same 20 calls.

| fixture | original baseline | Wave-3 wall median (min–max) | mel median | encoder median | decode median | total-phase median |
|---|---:|---:|---:|---:|---:|---:|
| short | 149.9 ms | **63.8 ms** (45.8–103.3) | 1.8 ms | 6.0 ms | 55.9 ms | 63.7 ms |
| medium | 243.6 ms | **185.7 ms** (125.5–280.2) | 5.3 ms | 10.0 ms | 170.0 ms | 185.6 ms |
| long | 438.8 ms | **352.8 ms** (255.8–713.8) | 12.0 ms | 32.0 ms | 309.0 ms | 352.7 ms |

All CUDA transcripts were byte-exact against the checked-in short, medium, and
long golden text. A separate fresh process transcribed all three exactly and
exited with status 0. The flash-attention kill-switch also remained byte-exact;
its 5+20 medians were 36.0 / 98.1 / 298.8 ms in a later diagnostic run that
included experimental decode replay, with encoder medians 6.6 / 11.0 / 33.1 ms
(short / medium / long). The encoder A/B is the relevant comparison: flash
attention is not slower at long `T`.

The earlier single-call 653 ms long result is not a steady-state encoder
regression: it included first-use/per-shape CUDA graph warmup and high decode
variance. In the controlled run, the final encoder median is 32.0 ms while the
serial host-mediated decode is 309.0 ms. The encoder change therefore works as
intended, but the requested absolute targets are not reached: short misses by
3.8 ms, medium by 65.7 ms, and long by 52.8 ms. Persisting the C++ prediction and
joint object lifetimes does not persist their graphs—the current
`PredictionNet::step` and `Joint::step_argmax` still call one-shot `run_graph`
on every serial TDT step. A prototype replay conversion reached 26.3 / 66.1 /
282.9 ms on CUDA, but was reverted because the CPU byte-exact gate could not be
completed within the available validation window. Device-resident/fused decode
remains the next optimization step.
