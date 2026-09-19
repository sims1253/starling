#!/usr/bin/env bash
# Run the packaged CUDA executable outside the build environment with only
# the documented runtime packages (RUNTIME.md): CUDA 13.3 runtime + cuBLAS +
# the userspace driver library + the base C/C++ runtime.
#
# Default (GPU-less, e.g. the release runner): the container has no NVIDIA
# kernel driver, so no CUDA device is initialized — but the driver library
# still has to LOAD for the binary to start, and the version/ABI metadata
# checks exercise the full loader path. The backend line names the archive's
# compiled backend flavor, not the runtime-selected device. This is a
# startup check, not inference evidence.
#
# --with-gpu (a machine whose docker has the NVIDIA runtime): the driver is
# injected by the container runtime (the documented provider) and the same
# checks run under it. Representative inference is verified separately on
# real hardware and recorded on the tracking issue.
set -euo pipefail
with_gpu=0
args=()
for a in "$@"; do
    if [[ "$a" == "--with-gpu" ]]; then with_gpu=1; else args+=("$a"); fi
done
if [[ ${#args[@]} != 3 ]]; then
    echo "Usage: $0 [--with-gpu] ARCHIVE VERSION ABI_VERSION" >&2
    exit 2
fi
archive=$(realpath "${args[0]}")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image=starling-release-cuda-runtime:ubuntu22.04
docker build --tag "$image" --file "$script_dir/Dockerfile.cuda" "$script_dir"
gpu_flags=()
if [[ $with_gpu == 1 ]]; then gpu_flags=(--gpus all); fi
docker run --rm --network none --read-only --tmpfs /tmp:exec "${gpu_flags[@]}" \
    --mount "type=bind,source=$archive,target=/release.tar.gz,readonly" \
    -i "$image" bash -s -- "${args[1]}" "${args[2]}" "$with_gpu" <<'CHECK'
set -euo pipefail
work=$(mktemp -d)
cd "$work"
tar -xzf /release.tar.gz
binary=starling-serve-linux-cuda
test -s RUNTIME.md
sha256sum --check "$binary.sha256"
# Report the complete transitive loader dependencies in the release job log.
ldd "./$binary" | tee dependencies.txt
if grep -Fq 'not found' dependencies.txt; then
    echo 'Unresolved runtime dependency' >&2
    exit 1
fi
version=$("./$binary" --version)
printf '%s\n' "$version"
grep -Fqx "starling-serve $1" <<< "$version"
# The backend line reflects the compiled-in registry + auto-preference, not
# a probed device: it reads "cuda" on a GPU-less container too. Device-level
# verification (cuInit, inference) is separate evidence on real hardware.
grep -Fqx "backend: cuda" <<< "$version"
abi=$("./$binary" --abi-version)
printf 'ABI: %s\n' "$abi"
test "$abi" = "$2"
if [[ "$3" == "1" ]]; then
    echo "driver-backed run (--with-gpu): the container runtime injected the host driver"
else
    echo "GPU-less run: startup metadata only; no device was initialized (expected)"
fi
CHECK
