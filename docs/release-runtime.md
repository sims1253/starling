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
Release preflight also compares the executing workflow's CUDA version with the
checked-out tag. For a manual release, select a workflow ref with the same version.

## Prerequisites

| Artifact | Runtime prerequisites | Verification before upload |
| --- | --- | --- |
| `linux-vulkan` | x86_64 Ubuntu 22.04 with `libstdc++6`, `libgomp1`, and `libvulkan1`; a vendor Vulkan driver and supported GPU for inference | Extracted archive, checksum, loader dependencies, version, and ABI in a fresh Ubuntu 22.04 container with only these runtime packages |
| `linux-cpu` | x86_64 Ubuntu 22.04 with `libstdc++6` and `libgomp1`; no GPU or driver required | Extracted archive, checksum, loader dependencies, version, backend, and ABI in a fresh Ubuntu 22.04 container with only these runtime packages |
| `linux-cuda` | x86_64 Linux compatible with the Ubuntu 22.04 build; CUDA 13.3 runtime and cuBLAS libraries, their dependencies, and a compatible NVIDIA driver | Version and ABI on the build runner only |
| `linux-rocm` | x86_64 Linux compatible with the Ubuntu 22.04 build; ROCm 7.2.4 HIP runtime, rocBLAS, hipBLAS, and their dependencies; a compatible AMD driver and GPU | Version and ABI on the build runner only |
| `windows-cuda` | x86_64 Windows; CUDA 13.3 runtime and cuBLAS DLLs, their dependencies, and a compatible NVIDIA driver; runtime DLL directories on `PATH` | Version and ABI on the build runner only |
| `windows-vulkan` | x86_64 Windows; Vulkan loader and vendor Vulkan driver | Version and ABI on the build runner only |
| `windows-cpu` | x86_64 Windows; no GPU or driver required | Version and ABI on the build runner only |
| `macos-metal` | Apple Silicon with macOS 14 or later; Metal supplied by macOS | Version and ABI on the build runner only |
| `macos-cpu` | Apple Silicon with macOS 14 or later; no GPU required | Version and ABI on the build runner only |

The Linux builds also use the system C/C++ runtime. The Windows executable uses
the static MSVC runtime; this does not remove dependencies of vendor DLLs.
CUDA's driver library comes from the installed NVIDIA driver. Install vendor
runtime packages through the vendor's supported installer or package repository,
so their transitive dependencies are installed too. Linux must be able to find
these libraries through its loader configuration or `LD_LIBRARY_PATH`.

The ROCm build installs the toolkit from the vendor's versioned `7.2.4`
repository (`https://repo.radeon.com/rocm/apt/7.2.4`, jammy `main`). ROCm 7.2.4
is the archive's runtime contract: run the `linux-rocm` archive against the
ROCm 7.2.4 HIP/BLAS runtime. ROCm archives have not been verified on a machine
without the development SDK.
The CUDA, Windows (Vulkan and CPU), and macOS (Metal and CPU) archives also
lack that separate check. Do not treat their build-runner metadata checks as a
clean-machine guarantee.

## Linux Vulkan archive check

From a checkout of the release tag, run:

```bash
scripts/release-runtime/check-linux-vulkan.sh \
  starling-serve-linux-vulkan.tar.gz 0.1.0 6
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
  starling-serve-linux-cpu.tar.gz 0.1.0 6
```

The check additionally requires `backend: cpu` in the `--version` output so a
misconfigured GPU build cannot ship under the CPU name. The same startup-only
scope as above applies.
