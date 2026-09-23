# Independent Parakeet CPU inference PoC

`standalone_parakeet` takes a 16 kHz mono PCM16 WAV and a Parakeet TDT
Q4_K_M GGUF, then emits token IDs and text. It does not include, link, or call
ggml. Its direct path owns GGUF v3 metadata and mmap access, mel extraction,
three convolutional subsampling stages, 24 Conformer layers with relative
attention, the two-layer prediction LSTM, and TDT greedy decoding. It uses
Q4_K, Q6_K, Q8_0, and F32 weights from the real model. The hot Q4/Q6/Q8
projections use direct AVX2 integer dot kernels. AArch64 targets with the dot
product extension use NEON dot kernels for Q4/Q6/Q8; AArch64 also has NEON
F32 dot and 2×2 matrix kernels. Other targets retain scalar fallbacks.
OpenMP divides output rows across CPU threads and parallelizes exact SiLU and
depthwise convolution. There is no general tensor graph or backend dispatch.
Cross builds omit `-march=native`; an AArch64 build can opt into dot-product
instructions with `-DPOC_ARM_DOTPROD=ON` when the target CPU supports them.

The default Q4 kernel reads GGUF nibbles in place and retains only decoded
scale metadata. `--pack-q4` trades much more memory for a packed layout; it
did not reliably improve the complete model on this host. The raw AVX2 Q4
kernel shares each decoded weight block across four audio frames and stores
only the eight activation sums used by its minimum correction. The first
subsampling convolution shares its nine input values across output channels.
The F32 matrix kernel computes two rows and two frames together, sharing
weight and input loads. The encoder keeps
one bounded cache of projected relative positions for repeated window
lengths (up to 32 MiB). `Encoder::run_window` also keeps one prior PCM window
and its *raw* log-mel features. It reuses only frames whose entire source
footprint lies in byte-identical overlapping PCM. Each window recomputes
CMVN. `Encoder::run` resets that audio cache for independent requests.

## CPU-only build and comparison

From the repository root:

```bash
cmake -S benchmarks/poc_q4_cpu -B build/poc-direct-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPOC_WITH_GGML_BASELINE=OFF
cmake --build build/poc-direct-cpu --target standalone_parakeet -j 4

cmake -S backends/native -B build/poc-native-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSTARLING_GGML_SHARED=ON \
  -DSTARLING_GGML_CUDA=OFF -DSTARLING_GGML_VULKAN=OFF \
  -DSTARLING_GGML_HIP=OFF -DSTARLING_GGML_METAL=OFF
cmake --build build/poc-native-cpu --target starling_ggml -j 4

taskset -c 0,2,4,6 python3 benchmarks/poc_q4_cpu/compare_parakeet_cpu.py \
  --gguf /path/to/parakeet-q4-k-m.gguf --wav /path/to/speech.wav \
  --library build/poc-native-cpu/libstarling_ggml.so \
  --direct build/poc-direct-cpu/standalone_parakeet --threads 4 --runs 3
```

An optional AVX2/FMA diagnostic build follows ggml's FP16 first-convolution
inputs, layer-norm reduction, vector SiLU, and Q4_K/Q6_K accumulator order:

```bash
cmake -S benchmarks/poc_q4_cpu -B build/poc-direct-cpu-reference -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DPOC_WITH_GGML_BASELINE=OFF \
  -DPOC_REFERENCE_NUMERICS=ON
cmake --build build/poc-direct-cpu-reference --target standalone_parakeet -j 4
```

This mode is for parity diagnosis. It requires AVX2/FMA, and it has not passed
the full recording parity gate. The default build retains the faster direct
matmul reductions and portable CPU fallbacks.

The comparison forces the baseline to CPU, checks the reported backend,
warms both engines, compares text and token IDs, and reports medians. It
reports blank-token differences separately; `--strict-ids` makes those a
failure. The standalone executable has no ggml or GPU library dependency
(`ldd build/poc-direct-cpu/standalone_parakeet`). Pin cores only if the
platform supports `taskset`.

To exercise the overlap cache on a WAV containing two full windows:

```bash
build/poc-direct-cpu/standalone_parakeet \
  /path/to/parakeet-q4-k-m.gguf /path/to/long-speech.wav 4 \
  --stream-window-s 12 --stream-advance-s 9
```

This checks cached against uncached mel, projected encoder output, and IDs
for the second window. The 12-second/9-second-advance test reused 297 of
1201 mel frames; all three comparisons were exact. This is a model-side
window cache, not yet a server or Android integration. Changing one sample
in the overlap caused zero reused frames and still matched a fresh frontend.

To compare a recording longer than a minute, with each engine loaded once in
its own CPU-only process:

```bash
taskset -c 0,2,4,6 python3 benchmarks/poc_q4_cpu/compare_stream_cpu.py \
  --gguf /path/to/parakeet-q4-k-m.gguf --wav /path/to/long-speech.wav \
  --library build/poc-native-cpu/libstarling_ggml.so \
  --direct build/poc-direct-cpu/standalone_parakeet \
  --threads 4 --window-s 12 --advance-s 9 --runs 2
```

This follows the server's finalized-window schedule: one 12 s window every
9 s, then a final tail. It omits speculative partial previews and word
stitching. The processes alternate order between rounds and exclude one warm
window from each timed session. Text differences make the command exit 1;
`--allow-text-mismatch` reports exploratory timings despite those differences.

The older `standalone_q4k` and `poc_q4_cpu` targets remain as isolated Q4_K
projection benchmarks. `poc_q4_cpu` links ggml **only** to calculate the CPU
reference result; `standalone_q4k` and `standalone_parakeet` do not.

## Measurements, 2026-09-23

AMD Ryzen 9 5900X, Parakeet 0.6B v3 Q4_K_M GGUF, CPU only, native Release
build. A separate training job was active, so times vary enough that these
numbers are directional. Each timing is a warm median in seconds; the
denominator in the ratio is the direct path.

| Audio | CPU threads | ggml | Direct | ggml/direct | Text |
| --- | ---: | ---: | ---: | ---: | --- |
| 2.5 s speech cut | 4 | 0.272 | 0.176 | 1.55× | match |
| 6.31 s speech | 1 | 1.792 | 1.182 | 1.52× | match |
| 6.31 s speech | 4 | 0.521 | 0.457 | 1.14× | match |
| 12 s speech window | 4 | 1.026 | 0.921 | 1.12× | match |

The 12-second row uses medians across three separate paired comparisons,
each with three warm inference runs. Their individual ratios ranged from
1.06× to 1.21×. The other rows are warm medians from one paired comparison.

For the 6.31-second four-thread run, peak RSS was about 765 MiB for direct
with raw Q4 metadata versus 785 MiB for the ggml baseline. `--pack-q4`
raised direct peak RSS to about 1,040 MiB without a stable full-model speed
gain. The direct and ggml **nonblank** token sequences matched on the three
timed speech windows and seven of eight additional real speech clips. One
real clip differed in punctuation after “away”; that difference also occurs
in the direct engine before these optimizations. Blank timing differed on
other clips, and final joint-projected encoder
output had about 0.0065 RMS difference against a 0.214 RMS reference. The
frontend used a maximum 4e-5 absolute difference on the 6.31-second clip.
These are transcription-parity results, not a proof of bit-exact internals.

In 20 alternating runs of an isolated Q4_K 1024×4096 projection at batch
150, the direct raw kernel's median fell from 3.97 ms to 3.13 ms after
four-frame weight sharing. The output was byte-identical. In six alternating
full-engine pairs, parallel depthwise convolution cut two-run subsampling
time from 0.240 s to 0.129 s and the warm full-inference median from 1.000 s
to 0.934 s. Exact parallel SiLU halved its measured wall time without
changing encoder bytes. A polynomial vector SiLU was faster again but changed
punctuation on one of eight real speech clips, so it was rejected. The final
engine matched the previous exact path's IDs and text on all eight clips.
In eight alternating full-engine pairs, the 2×2 F32 tile cut cumulative F32
matmul time from 0.341 s to 0.257 s with byte-identical encoder output.

Callgrind on the isolated 1024×4096 Q4 projection at batch 150 and one CPU
thread counted 1.78 billion instructions and 32.97 million conditional
branches before the four-frame tile, versus 1.29 billion instructions and
16.65 million branches after it. A full 1.64-second clip at one thread went
from 5.22 to 4.88 billion simulated instructions and from 238 to 189 million
simulated L1 data misses after the F32 tile. Simulated last-level data misses
were effectively unchanged. Callgrind counters provide a less clock-sensitive
comparison; they are not estimates of phone energy.

### Longer recordings, 2026-09-24

The 74.35 s `tests/fixtures/long.wav` contains eight transcription requests:
seven full 12 s windows and the final tail. A 143.59 s fixture joined that
recording with the 69.24 s VoxPopuli `clip_00000.wav` from the local test
corpus; it made 15 full windows and a final tail. Both engines used the same
Parakeet Q4_K_M GGUF, four pinned CPU threads, separate processes, and one
excluded warm window. Two paired rounds alternated execution order. The
training job was still active, so wall-time variation remains material.

| Audio | ggml session, s | Direct session, s | Paired wall ratios | CPU-time ratio |
| --- | ---: | ---: | --- | ---: |
| 74.35 s, 8 windows | 7.93 | 7.68 | 1.12×, 0.96× | 1.09× |
| 143.59 s, 16 windows | 15.49 | 13.37 | 1.15×, 1.17× | 1.20× |

Session values are medians of the two rounds; each ratio is ggml divided by
direct for one paired round. CPU-time ratio compares aggregate process CPU
time and is only a scheduling-resistant work indicator, not an energy
measurement. These finalized-window sessions do not demonstrate a 1.5×
speedup. The earlier 2.5 s clip and single-thread 6.31 s clip did reach about
1.5×, but they are different operating points.

**Parity gate failed on one window.** At 36–48 s in the long fixture, ggml
transcribes “observed Phoebe, turning away her eyes.” while direct ends after
“observed.” The other seven windows in the 74.35 s recording match text; the
same mismatch is one of 16 windows in the joined recording. ggml's text is
stable across 1, 2, 4, and 8 CPU threads; direct's shortened text is stable
at 1 and 4. Feeding ggml's projected encoder tensor into the independent
decoder recovers ggml's complete text, locating the discrepancy in the
encoder. For this window, mel RMS difference is 1.5e-6, but projected encoder
RMS difference is 0.0183 against ggml's 0.1755 RMS output. `--input-encoder`
accepts the same frame-count-plus-F32 format as `--dump-encoder` for this
isolation check. A temporary stage dump found a difference already at the
subsampling output, which then propagated through the 24 encoder layers. The
direct engine needs encoder parity work before replacing the ggml serving path.

A two-frame Q6 tile was also tried. It preserved encoder bytes, but its
benefit across 12 s native runs was inconsistent and a 0.5 s Callgrind run
counted about 35 million more instructions in the Q6 path. It was removed.
Keeping Q4 floating-point accumulators across a whole row, as ggml does,
also failed: it barely changed the subsampling discrepancy, made Q4 about
9% slower in paired 12 s runs, and changed that clip's transcript. It was
removed; the final 12 s encoder dump is byte-identical to the previous exact
direct kernel.

There is no MOSS Q4 GGUF artifact in the workspace or in the searched local
model locations, so this standalone engine currently implements Parakeet
only, as permitted for this PoC. It has not been integrated into the serving
or Android path. The AArch64 dot-product intrinsics passed an isolated
cross-compilation syntax check, but the full engine has not been built or run
on an ARM device. Phone latency, energy,
and thermal behavior remain unmeasured.

### Numerical diagnosis and two-minute gate

On the saved 36–48 s failing window, feeding ggml's encoder result to the
independent decoder restores its complete transcript. With *identical* layer
input, the diagnostic Q4_K and Q6_K accumulators, AVX2 SiLU, and layer norm
each reproduce their ggml CPU component output byte for byte. The ordinary
Q4 projection differs by roughly 1e-6 RMS on that input; a later quantized
projection can amplify such a small difference into a different token. The
full diagnostic mode fixes this window's nonblank tokens and text.

The 148.70 s test repeats `long.wav` twice and makes 16 full 12 s windows
plus a tail at a 9 s advance. With the experimental vector SiLU fast path,
two isolated CPU pairs gave ggml/direct ratios of 1.260× and 1.270×, but
window 4 still differed. The smaller layer-norm-plus-Q6 diagnostic gave
1.225× and 1.223× with mismatches at windows 7 and 12. The full reference
mode gave 1.013× in one pair and still differed at window 6. These are
different numeric modes, not a speed progression. The training run on this
host makes wall-time ratios directional; CPU time was recorded by
`compare_stream_cpu.py` as a second work indicator. No mode has passed the
two-minute text gate.

SIMD Q6 byte expansion and an eight-frame Q4 tile were tried and removed.
The Q6 expansion reduced Callgrind's 0.5 s simulated instruction count by
67 million, but not the 12 s CPU wall time. The eight-frame Q4 tile preserved
encoder bytes and reduced one whole-process Callgrind count; its Q4 function
count and paired 12 s process CPU time both increased. Those results do not
support either as a power or latency improvement.
