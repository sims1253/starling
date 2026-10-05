# Release runtime requirements

Release archives contain the executable, its SHA-256 checksum, and this file
(as `RUNTIME.md`). They do not bundle accelerator libraries, GPU drivers, or
models. `BUILD_SHARED_LIBS=OFF` links the project libraries into the executable;
it does not make CUDA or ROCm libraries static.

The release workflow's `CUDA_VERSION` sets the Windows toolkit patch and the
Linux toolkit series. Linux installs that series' metapackage, which can select a
later patch. Runtime guidance names the shared major/minor series. When changing
it, update the CUDA requirements here, in the native serving guide, and in the
release body; `scripts/release-runtime/check-contract.py` checks their agreement.
The pinned ROCm apt repository works the same way: the versioned URL in the
workflow is the source of truth, and the checker requires the `linux-rocm` row
here, the serving guide, and the release body to restate that exact version.
Release preflight also compares the executing workflow's CUDA version with the
checked-out tag. For a manual release, select a workflow ref with the same version.

## Prerequisites

| Artifact | Runtime prerequisites | Verification before upload |
| --- | --- | --- |
| `linux-vulkan` | x86_64 Ubuntu 22.04 with `libstdc++6`, `libgomp1`, and `libvulkan1`; a vendor Vulkan driver and supported GPU for inference | Extracted archive, checksum, loader dependencies, version, and ABI in a fresh Ubuntu 22.04 container with only these runtime packages |
| `linux-cpu` | x86_64 Ubuntu 22.04 with `libstdc++6` and `libgomp1`; no GPU or driver required | Extracted archive, checksum, loader dependencies, version, backend, and ABI in a fresh Ubuntu 22.04 container with only these runtime packages |
| `linux-cuda` | x86_64 Linux compatible with the Ubuntu 22.04 build; CUDA 13.3 runtime and cuBLAS libraries, their dependencies, and a compatible NVIDIA driver | Extracted archive, checksum, loader dependencies, version, backend, and ABI in a fresh Ubuntu 22.04 container with only these runtime packages and the userspace driver library (startup only; no GPU) |
| `linux-rocm` | x86_64 Linux compatible with the Ubuntu 22.04 build; ROCm 7.2.4 HIP runtime, rocBLAS, hipBLAS, and their dependencies; a compatible AMD driver and GPU | Version and ABI on the build runner only |
| `windows-cuda` | x86_64 Windows; the Microsoft Visual C++ Redistributable (x64); the CUDA 13.3 runtime's cuBLAS DLLs (`cublas64_13.dll`, `cublasLt64_13.dll`) with their directory on `PATH`; a compatible NVIDIA driver. The CUDA runtime library itself is linked statically | Version and ABI on the build runner only |
| `windows-vulkan` | x86_64 Windows; the Microsoft Visual C++ Redistributable (x64); Vulkan loader and vendor Vulkan driver | Version and ABI on the build runner only |
| `windows-cpu` | x86_64 Windows; the Microsoft Visual C++ Redistributable (x64); no GPU or driver required | Version and ABI on the build runner only |
| `macos-metal` | Apple Silicon with macOS 14 or later; Metal supplied by macOS | Version and ABI on the build runner only |
| `macos-cpu` | Apple Silicon with macOS 14 or later; no GPU required | Version and ABI on the build runner only |

The Linux builds also use the system C/C++ runtime. The Windows executables use
the static MSVC runtime, but they also import `vcomp140.dll`, the MSVC OpenMP
runtime behind ggml's CPU threading, which the static runtime does not cover.
It comes with the Microsoft Visual C++ 2015-2022 Redistributable (x64), the same
package the desktop app needs; without it the executable does not start.
The static runtime also does not remove dependencies of vendor DLLs.
CUDA's driver library comes from the installed NVIDIA driver.

The Windows CUDA executable delay-loads cuBLAS so that `--version` and
`--abi-version` work without it. When it selects a CUDA device it loads
`cublas64_13.dll` through the normal DLL search order (the executable's
directory, the system directories, then `PATH`). If that fails, the model load
fails with an error naming the DLL: with the default eager load the server
prints it and exits; with `--no-eager-load` the server keeps running, answers
transcriptions with 503 "model not loaded", and `/health` reports the error as
`load_error`. NVIDIA's redistributable `libcublas` archive for CUDA 13.3
provides both cuBLAS DLLs. Install vendor
runtime packages through the vendor's supported installer or package repository,
so their transitive dependencies are installed too. Linux must be able to find
these libraries through its loader configuration or `LD_LIBRARY_PATH`.

The ROCm build installs the toolkit from the vendor's versioned `7.2.4`
repository (`https://repo.radeon.com/rocm/apt/7.2.4`, jammy `main`). ROCm 7.2.4
is the archive's runtime contract: run the `linux-rocm` archive against the
ROCm 7.2.4 HIP/BLAS runtime. ROCm archives have not been verified on a machine
without the development SDK.

The release workflow checks the Windows and macOS archives on their build
runners only. Do not treat those metadata checks as a clean-machine
guarantee; the Windows archives have been checked separately on one machine
(see [Windows archive check](#windows-archive-check) and
[Hardware verification](#hardware-verification)), the macOS archives have
not. The Linux CUDA archive has the same style of fresh-container startup
check as Linux Vulkan/CPU (see below); like them, the per-release check does
not verify GPU inference.

## Desktop bundled engines

The Linux and Windows desktop archives bundle exactly the `vulkan` and `cpu`
engines (starling-serve binaries) inside an `engines/` directory; the
bundle's preference order is `vulkan` first, so the app picks it when a
Vulkan driver is present and falls back to `cpu`. Their runtime
prerequisites are the rows above: `linux-vulkan` and `linux-cpu` for the
Linux archive, `windows-vulkan` and `windows-cpu` for the Windows archive.
Each archive carries the engines' `RUNTIME.md` and checksums; `engines.json`
records the shared version and ABI. CUDA is not bundled; it stays a separate
standalone server download for headless and advanced use.

## Linux Vulkan archive check

From a checkout of the release tag, run:

```bash
scripts/release-runtime/check-linux-vulkan.sh \
  starling-serve-linux-vulkan.tar.gz 0.1.0 8
```

Replace the version and ABI with the values expected for that tag. Docker builds
an Ubuntu 22.04 image with the packages listed above, then runs the archive check
with networking disabled. Only the archive is mounted into the container. No SDK,
build directory, host library paths, or GPU devices are supplied. The check
extracts the archive, verifies its checksum, reports loader dependencies, and
requires the expected version and ABI. Any missing library fails the check.

This check covers executable startup. It does not load a model or prove that a
GPU can run inference. Before claiming support for a GPU, run a representative
model on that hardware and record the artifact checksum, OS, driver version,
runtime version, model, and result. That hardware validation remains separate.

## Linux CPU archive check

Same procedure against the CPU image, which installs only `libstdc++6` and
`libgomp1`:

```bash
scripts/release-runtime/check-linux-cpu.sh \
  starling-serve-linux-cpu.tar.gz 0.1.0 8
```

The check additionally requires `backend: cpu` in the `--version` output so a
misconfigured GPU build cannot ship under the CPU name. The same startup-only
scope as above applies.

## Linux CUDA archive check

From a checkout of the release tag, run:

```bash
scripts/release-runtime/check-linux-cuda.sh \
  starling-serve-linux-cuda.tar.gz 0.1.0 8
```

Replace the version and ABI with the values expected for that tag. Docker
builds an Ubuntu 22.04 image with the documented runtime packages — the CUDA
13.3 runtime and cuBLAS from the vendor repository, the base C/C++ runtime,
and the userspace driver library (`libnvidia-compute-610` from the vendor
repository provides `libcuda.so.1`, the loader-resolution half of the
documented "compatible NVIDIA driver" prerequisite; an installed driver
supplies the same soname) — then runs the archive check with networking
disabled and nothing but the archive mounted. No CUDA SDK, build directory,
or host library paths are supplied.

The check verifies the checksum, requires every loader dependency to resolve,
and gates the version, `backend: cuda`, and ABI. The `--version` backend line
names the compiled backend flavor, not a probed device. The default mode is
GPU-less — exactly what the release runner executes after packaging, before
upload; on a machine whose container runtime can inject the NVIDIA driver,
`--with-gpu` reruns the same checks under it. Either way this is a startup
check: it does not initialize a CUDA device or run inference.

On a machine with an NVIDIA GPU, `--infer` adds representative inference to
the driver-backed run:

```bash
scripts/release-runtime/check-linux-cuda.sh --with-gpu \
  --infer parakeet-tdt-0.6b-v3-q8_0.gguf jfk.wav "<expected transcript>" \
  starling-serve-linux-cuda.tar.gz 0.1.0 8
```

The Parakeet GGUF and a 16 kHz mono PCM16 WAV are mounted read-only; the
container still has no network, SDK, or host library paths. The packaged
server loads the model, `/health` must report a CUDA device, and one
`POST /v1/audio/transcriptions` must return exactly the expected transcript
with no accelerator-rejected graph node (#184). Results are recorded under
[Hardware verification](#hardware-verification).

## Windows archive check

`scripts/release-runtime/check-windows-archive.ps1` (Windows PowerShell 5.1)
runs a Windows archive outside its build environment. It extracts the archive
into a fresh directory, verifies the checksum, and runs every process with a
`PATH` reduced to the Windows system directories plus `-RuntimePath`, so a
CUDA archive resolves cuBLAS from the documented runtime directories rather
than an installed toolkit. It then gates the version, backend flavor, and ABI:

```powershell
powershell -ExecutionPolicy Bypass -File check-windows-archive.ps1 `
  -Archive starling-serve-windows-cuda.zip -Version 0.1.0 -Abi 8 `
  -RuntimePath C:\cuda-13.3\libcublas\bin\x64 `
  -Gguf parakeet-tdt-0.6b-v3-q8_0.gguf -Audio jfk.wav -Expected "<expected transcript>"
```

With `-Gguf`, `-Audio`, and `-Expected` it also runs representative inference:
`/health` must report a device of the archive's backend (`CUDA0`, `Vulkan0`, or
`CPU`), the transcript must match exactly, and the server log must contain no
accelerator-rejected node. It refuses a port that is already in use, requires
the listening socket to belong to the server it started, and bounds the
transcription request. It lists the runtime and driver DLLs the server
loaded, and for CUDA requires cuBLAS to come from `-RuntimePath`. Installed SDKs
elsewhere on the machine are not removed, so this is a reduced-`PATH` check,
not a clean-VM check. The release workflow does not run it; results are
recorded below.

## Hardware verification

Startup checks do not prove that inference works. This ledger records which
artifacts have run representative inference on real hardware: the Parakeet TDT
0.6B v3 Q8_0 GGUF (`scholzmx/parakeet-tdt-0.6b-v3-gguf`, sha256
`cbab40be5510f86f825ccb19bdea0876938358a2266a24a1ab8f1fccf0759922`) on the
11 s JFK clip (16 kHz mono PCM16, sha256
`7fa5360b8f318ab7d541c89b54da36eabb47636d32d89ec02ffc3e6ca046dc9e`) through
`POST /v1/audio/transcriptions`, requiring the exact transcript on the device
the archive targets. Only Parakeet was exercised; other models are not covered
by this ledger. The release workflow does not re-run these checks, so each
entry is for the build named in it. "E25" is the Experimental 25 prerelease
(master `74a8fe3`, built by this workflow in experimental mode, CUDA SM 120
only). Checksums, versions, procedures, and expected versus actual results
for the 2026-10-05 entries are in the
[evidence record](https://github.com/sims1253/starling/blob/master/docs/evidence/release-runtime-2026-10-05/README.md)
(tracking issue #57).

| Artifact | Representative inference | Hardware and runtime | Not covered |
| --- | --- | --- | --- |
| `linux-cuda` | Verified (2026-10-05): E25 archive and a local release-recipe build (SM 75-120, CUDA 13.3) | RTX 5090 (SM 120) through WSL2: Windows driver 610.88, Ubuntu 22.04 container with only the documented runtime packages | Bare-metal Linux NVIDIA driver; SM 75-90 GPUs |
| `linux-vulkan` | Verified (#76) | AMD Radeon Graphics (RADV RENOIR), Mesa 26.2.2 and 23.2.1 | NVIDIA and Intel Vulkan drivers |
| `linux-cpu` | Verified (2026-10-05): E25 archive | AMD Ryzen 9 5900X, Ubuntu 22.04 (WSL2) | Non-x86_64 hosts are not a release target |
| `linux-rocm` | Not verified | No ROCm hardware available | Everything; only the build runner's version and ABI checks ran |
| `windows-cuda` | Verified (2026-10-05): E25 archive | RTX 5090 (SM 120), Windows 11 build 26200, driver 610.88; cuBLAS 13.5.1 from NVIDIA's CUDA 13.3.0 redistributables; reduced `PATH` | SM 75-90 GPUs; a clean Windows install |
| `windows-vulkan` | Verified on a local build only (2026-10-05). The E25 archive aborted on the first transcription on NVIDIA (misaligned duration-argmax source); the fix is not yet in a workflow-built archive | RTX 5090, NVIDIA Vulkan driver 610.88 (Vulkan 1.4), Windows 11 build 26200 | AMD and Intel Vulkan drivers; the workflow-built archive with the fix |
| `windows-cpu` | Verified (2026-10-05): E25 archive | AMD Ryzen 9 5900X, Windows 11 build 26200, reduced `PATH` | A clean Windows install |
| `macos-metal` | Not verified | No Apple hardware available | Everything; only the build runner's version and ABI checks ran |
| `macos-cpu` | Not verified | No Apple hardware available | Everything; only the build runner's version and ABI checks ran |
