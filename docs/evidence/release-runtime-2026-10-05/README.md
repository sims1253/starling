# Release runtime hardware verification, 2026-10-05

Evidence behind the [Hardware verification](../../release-runtime.md#hardware-verification)
ledger entries dated 2026-10-05 (issues #57 and #184). Every run was on one
machine; no release was published and no release workflow was dispatched.

## Machine

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 9 5900X 12-Core |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, 32 GiB |
| Windows | Windows 11 Pro Education 10.0.26200 |
| NVIDIA driver | 610.88 (Windows driver version 32.0.16.1088) |
| VC++ redistributable | Microsoft Visual C++ 2015-2022 x64 14.51.36247 (`vcomp140.dll` in System32) |
| Linux | WSL2, Ubuntu 22.04.5, kernel 6.18.33.2-microsoft-standard-WSL2; GPU access through WSL GPU paravirtualization, not a native Linux NVIDIA kernel driver |

## Inputs

| Input | Value |
| --- | --- |
| Model | `scholzmx/parakeet-tdt-0.6b-v3-gguf` revision `96402b32bd374742aa1da3c66af30aa64cea3fdb`, `parakeet-tdt-0.6b-v3-q8_0.gguf`, sha256 `cbab40be5510f86f825ccb19bdea0876938358a2266a24a1ab8f1fccf0759922` |
| Clip | `openai/whisper` `tests/jfk.flac` (sha256 `63a4b1e4c1dc655ac70961ffbf518acd249df237e5a0152faae9a4a836949715`) converted with ffmpeg 4.4.2 `-ar 16000 -ac 1 -c:a pcm_s16le` to `jfk.wav`, sha256 `7fa5360b8f318ab7d541c89b54da36eabb47636d32d89ec02ffc3e6ca046dc9e` (11 s) |
| Longer clips | `-t 3` (sha256 `d0f0dc7f2451d8364c1dd7c70a7eba7165ce6a98d46b473e8841d6a3a4246a47`), `-stream_loop 3` (44 s, `591312bd845f63ca30f67d308369d97ab118e7a4b1ec1716faf7f388e8ade575`), `-stream_loop 5` (66 s, `41af2de49bc302b5450e00f5835a8074de56fad1bdc7c013009cb223d5abb80e`) of `jfk.wav`; the 66 s clip exceeds 512 encoder frames and uses the K=64 TDT graph |
| Expected | `And so, my fellow Americans, ask not what your country can do for you, ask what you can do for your country.` |

## Artifacts

Workflow-built archives come from the **Experimental 25** prerelease (tag
`experimental-25-74a8fe3909ca`, master `74a8fe3909ca9b3c223675304f63efc378f195a6`,
[run 37078967790](https://github.com/sims1253/starling/actions/runs/37078967790)),
built by `release-starling-serve.yml` in experimental mode (CUDA SM 120 only);
each archive matched the release's `SHA256SUMS.txt`. Local builds are from
master `1060be8` or the commits of the change that added this record.

| Label | Archive / executable sha256 | Build |
| --- | --- | --- |
| E25 `linux-cuda` | archive `4c8d73cbb9896a34b031105df557a69aae244b55c05e38ed4ad928174ea13eb1`, exe `86dbbec38743ac46a7646419ab94f0425566bd15efcd9f46bf2b6b4b4244be09` | workflow, CUDA 13.3, SM 120 |
| E25 `linux-cpu` | archive `b924ec27f3ebc9c2a0e69d762e810adfe53fa9797f4387f4316bc26543d31f9b`, exe `3bb434e246d1af4cc821b408350620dddfd3338b1bdab7e4ce4eeac7d3cca989` | workflow |
| E25 `windows-cuda` | archive `af363806360a2727b5a6bf5d78d16d469f9c41446692768a4910ab9d95abb143`, exe `a40c93b161bb69d6876ef02b085502b494c2427cee6d1ab156dd0772a569442c` | workflow, CUDA 13.3, SM 120 |
| E25 `windows-vulkan` | archive `b163e78fc3e5f4896e1058dd7bbaa45d665c884231726321520c44db38ec67d9`, exe `8b0c2371477fd7bf408267f49c052d5408fcb3568d0ca7bcbcd8efe9087f4c88` | workflow |
| E25 `windows-cpu` | archive `f4a3131c2fb277318804641673da85c191546aa9d1fd28cdac32c4515833406a`, exe `542892603ec53e96f86e517ac67310695caa9df176f59537556b69c00ec705e9` | workflow |
| L1 `linux-cuda` (#184 repro) | exe `26df43e3b55006eda45982917a9440c6b2304306c4f9b8a945adf05d4caa4a9d` (unstripped) | master `1060be8`, the `linux-cuda` job's CMake flags, SM 75;80;86;89;90;120, host nvcc 13.0.88, GCC 11.4 |
| L2 `linux-cuda` (fix) | exe `137bfa0d6314c62fc07871226f2d7ecc812a5c2f8ddd6d185f86a9916a8d3ea2` (unstripped) | as L1 with the duration-argmax fix |
| L3 `linux-cuda` (release recipe, fix) | archive `4b9a8e55f6d0ac3765d18d32259a306584919f1e837e170382ffd5f6ee5e7dc9`, exe `3abffae811fc206a46afbbadc7ee5d950dab3c84031635e5d5d7f16467a8a319` | `ubuntu:22.04` container with `cuda-toolkit-13-3` 13.3.1 (nvcc 13.3.73), GCC 11.4, SM 75..120, stripped and packaged as in the workflow |
| L4 `windows-vulkan` (master) | exe `d5b5182f3b6309a8bc607409e202c40142f5c00119b62035003fbf8a9982bd74` | master `1060be8`, the `windows-vulkan` job's CMake flags, VS 2022 17.14 (MSVC 14.44), Vulkan SDK 1.4.341.1 (the workflow uses VS 2026 and SDK 1.4.357.0) |
| L5 `windows-vulkan` (fix) | exe `531fe3fce3273d1275ceb0e4197b71538c962aeddf764abe5546349413e3cea7` | as L4 with the duration-argmax fix |
| L6 `windows-cuda` (fix + cuBLAS probe) | exe `511dc530d51f4471d7b9ca05aaa36a301cbc3b26420300d0f8733b5051a01042` | the `windows-cuda` job's CMake and delay-load flags, VS 2022 17.14, CUDA 13.2 toolkit, SM 120 |

Windows CUDA runtime: NVIDIA's CUDA 13.3.0 redistributables
(`redistrib_13.3.0.json`): `cuda_cudart-windows-x86_64-13.3.29-archive.zip`
(sha256 `1feb7dd266813ffe8dbc24e115183a5ac35a4795c8d34aca0df85ab616b64d9c`)
and `libcublas-windows-x86_64-13.5.1.27-archive.zip` (sha256
`c946e1c825e05895747a95ed4fee18030b08052c09783b9b7b19818fd2e31f58`), both
matching the manifest.

Linux CUDA runtime container: `scripts/release-runtime/Dockerfile.cuda`
(`cuda-cudart-13-3` 13.3.29, `libcublas-13-3` 13.6.0.2,
`libnvidia-compute-610` 610.57.04). Under `--gpus all` the container runtime
bind-mounts the host's WSL `libcuda.so.1.1` (sha256
`21b9179c622a519d339d3387bca37179a26b9f65548c8524a41c5c4ce43671f6`) over the
image's `libcuda.so.610.57.04`, so the host driver serves the GPU.

## Results

Every inference row posted the 11 s clip to `/v1/audio/transcriptions` with
`STARLING_SCHED_DEBUG=1` and required the expected transcript exactly.

| Artifact | Procedure | Device | Actual |
| --- | --- | --- | --- |
| L1 | issue #184 repro on the host | `CUDA0` | exact transcript, HTTP 200, 0 `[sched-dbg]` lines, no assert, server alive; `tests/test_native_serve.py` real-model suite 81 passed, 0 failed |
| L2 | `tests/test_native_serve.py` real-model suite | `CUDA0` | 81 passed, 0 failed |
| E25 `linux-cuda` | `check-linux-cuda.sh --with-gpu --infer` | `CUDA0` | pass: checksum, all loader dependencies resolved, version/backend/ABI, exact transcript |
| L3 | `check-linux-cuda.sh` (GPU-less), then `--with-gpu --infer` | `CUDA0` | both pass; wrong expected text fails (exit 1) |
| E25 `linux-cuda` vs L2 | 3 s, 11 s, 44 s, 66 s clips on the host | `CUDA0` | transcripts identical between the two builds; 0 `[sched-dbg]` |
| E25 `linux-cpu` | host run, checksum verified | `CPU` | exact transcript |
| E25 `windows-cuda` | `check-windows-archive.ps1 -RuntimePath <cudart and cuBLAS redist bin>` | `CUDA0` | pass. Loaded `cublas64_13.dll` and `cublasLt64_13.dll` 6.14.11.1351 from the redistributable, `nvcuda.dll` 32.0.16.1088 from the driver; `cudart64_13.dll` was not loaded (the runtime is linked statically) |
| E25 `windows-cuda` | system directories only on `PATH` | `CUDA0` | the model loads and `/health` reports `loaded:true`; the process then dies without a message on the first transcription |
| E25 `windows-vulkan` | `check-windows-archive.ps1` | `Vulkan0` | **fail**: the first transcription aborts with `ggml-vulkan.cpp:2539: GGML_ASSERT(!src0 \|\| get_misalign_bytes(ctx, src0) == 0)` |
| E25 `windows-cpu` | `check-windows-archive.ps1` | `CPU` | pass |
| L4 | `check-windows-archive.ps1` | `Vulkan0` | **fail**: the same assert |
| L5 | `check-windows-archive.ps1`; then the 3 s, 11 s, 44 s, 66 s clips | `Vulkan0` | pass (NVIDIA `nvoglv64.dll` 32.0.16.1088, `vulkan-1.dll` 1.4.341.0); all four transcripts identical to the CUDA ones, 0 `[sched-dbg]`, server alive |
| L6 | `check-windows-archive.ps1 -RuntimePath <cuBLAS redist bin>` | `CUDA0` | pass, cuBLAS from the redistributable |
| L6 | system directories only, eager load | — | `load FAILED: CUDA device CUDA0 needs cublas64_13.dll (cuBLAS), which could not be loaded; …`, then exit |
| L6 | system directories only, `--no-eager-load` | — | transcription answers 503 `model not loaded`; `/health.load_error` carries the same message; process alive |

`dumpbin /dependents` on the E25 Windows executables: CPU imports `KERNEL32`,
`ADVAPI32`, `WS2_32`, `VCOMP140`; Vulkan adds `vulkan-1.dll`; CUDA has the CPU
set as eager imports and `cublas64_13.dll` as its only delay-load import. The
host has the VC++ redistributable installed, so the missing-`vcomp140.dll`
case was not run.

## Not covered

- `linux-rocm`, `macos-metal`, `macos-cpu`: no hardware.
- Linux CUDA with a native Linux NVIDIA kernel driver; CUDA GPUs other than SM 120.
- Vulkan on NVIDIA or Intel under Linux, and on AMD or Intel under Windows.
- A clean Windows installation (no VC++ redistributable, no SDKs).
- Models other than Parakeet TDT 0.6B v3 Q8_0.
- A workflow-built `windows-vulkan` archive that contains the fix.
