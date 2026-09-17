#!/usr/bin/env bash
# Run the packaged CPU executable without the build runner's toolchain.
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "Usage: $0 ARCHIVE VERSION ABI_VERSION" >&2
    exit 2
fi
archive=$(realpath "$1")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image=starling-release-cpu-runtime:ubuntu22.04
docker build --tag "$image" --file "$script_dir/Dockerfile.cpu" "$script_dir"
docker run --rm --network none --read-only --tmpfs /tmp:exec \
    --mount "type=bind,source=$archive,target=/release.tar.gz,readonly" \
    -i "$image" bash -s -- "$2" "$3" <<'CHECK'
set -euo pipefail
work=$(mktemp -d)
cd "$work"
tar -xzf /release.tar.gz
binary=starling-serve-linux-cpu
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
grep -Fqx "backend: cpu" <<< "$version"
abi=$("./$binary" --abi-version)
printf 'ABI: %s\n' "$abi"
test "$abi" = "$2"
CHECK
