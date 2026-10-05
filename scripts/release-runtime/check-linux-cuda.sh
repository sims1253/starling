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
# checks run under it.
#
# --infer GGUF AUDIO EXPECTED_TEXT (requires --with-gpu): representative
# inference on the host GPU. The Parakeet GGUF and a 16 kHz mono PCM16 WAV
# are mounted read-only next to the archive; the packaged server loads the
# model, /health must report a CUDA device, and one POST
# /v1/audio/transcriptions must return exactly EXPECTED_TEXT with no
# accelerator-rejected node (STARLING_SCHED_DEBUG=1, #184). The container
# still has no network, no SDK, and no host library paths; the HTTP client
# is bash's /dev/tcp, so no package beyond the documented runtime is added.
set -euo pipefail
with_gpu=0
infer=()
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-gpu) with_gpu=1; shift ;;
        --infer)
            if [[ $# -lt 4 ]]; then
                echo "--infer needs GGUF AUDIO EXPECTED_TEXT" >&2
                exit 2
            fi
            infer=("$2" "$3" "$4"); shift 4 ;;
        *) args+=("$1"); shift ;;
    esac
done
if [[ ${#args[@]} != 3 ]]; then
    echo "Usage: $0 [--with-gpu] [--infer GGUF AUDIO EXPECTED_TEXT] ARCHIVE VERSION ABI_VERSION" >&2
    exit 2
fi
if [[ ${#infer[@]} -gt 0 && $with_gpu != 1 ]]; then
    # A GPU-less container auto-selects the CPU backend, so a transcript
    # from it would say nothing about CUDA.
    echo "--infer requires --with-gpu" >&2
    exit 2
fi
archive=$(realpath "${args[0]}")
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
image=starling-release-cuda-runtime:ubuntu22.04
docker build --tag "$image" --file "$script_dir/Dockerfile.cuda" "$script_dir"
run_flags=()
if [[ $with_gpu == 1 ]]; then run_flags+=(--gpus all); fi
expected=""
if [[ ${#infer[@]} -gt 0 ]]; then
    run_flags+=(
        --mount "type=bind,source=$(realpath "${infer[0]}"),target=/model.gguf,readonly"
        --mount "type=bind,source=$(realpath "${infer[1]}"),target=/audio.wav,readonly"
    )
    expected=${infer[2]}
fi
docker run --rm --network none --read-only --tmpfs /tmp:exec "${run_flags[@]}" \
    --mount "type=bind,source=$archive,target=/release.tar.gz,readonly" \
    -i "$image" bash -s -- "${args[1]}" "${args[2]}" "$with_gpu" "$expected" <<'CHECK'
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
# verification is the --infer mode below.
grep -Fqx "backend: cuda" <<< "$version"
abi=$("./$binary" --abi-version)
printf 'ABI: %s\n' "$abi"
test "$abi" = "$2"
if [[ "$3" == "1" ]]; then
    echo "driver-backed run (--with-gpu): the container runtime injected the host driver"
else
    echo "GPU-less run: startup metadata only; no device was initialized (expected)"
fi
expected=$4
if [[ -z "$expected" ]]; then
    exit 0
fi

# ---- representative inference (--infer) ----
port=18187
# One HTTP/1.1 exchange over bash's /dev/tcp: METHOD PATH [BODY_FILE TYPE].
http() {
    exec 3<>"/dev/tcp/127.0.0.1/$port" || return 1
    if [[ $# -gt 2 ]]; then
        printf '%s %s HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: %s\r\nContent-Length: %s\r\nConnection: close\r\n\r\n' \
            "$1" "$2" "$4" "$(stat -c %s "$3")" >&3
        cat "$3" >&3
    else
        printf '%s %s HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n' "$1" "$2" >&3
    fi
    cat <&3
    exec 3<&-
}
STARLING_SCHED_DEBUG=1 "./$binary" --model parakeet --gguf /model.gguf \
    --host 127.0.0.1 --port "$port" > server.log 2>&1 &
server=$!
stop_server() {
    kill "$server" 2>/dev/null || true
    wait "$server" 2>/dev/null || true
}
trap stop_server EXIT
health=""
for _ in $(seq 600); do
    if ! kill -0 "$server" 2>/dev/null; then
        cat server.log >&2
        echo 'Server exited before the model loaded' >&2
        exit 1
    fi
    health=$(http GET /health 2>/dev/null | tail -n 1) || true
    if grep -Fq '"loaded":true' <<< "$health"; then break; fi
    sleep 0.5
done
printf 'health: %s\n' "$health"
grep -Fq '"loaded":true' <<< "$health" || { cat server.log >&2; echo 'Model did not load' >&2; exit 1; }
# /health names the runtime-selected device (CUDA0, ...), not the build flavor.
grep -Eq '"backend":"CUDA[0-9]+"' <<< "$health" || {
    echo 'The server did not select a CUDA device' >&2
    exit 1
}
boundary=starling-runtime-check
{
    printf -- '--%s\r\nContent-Disposition: form-data; name="model"\r\n\r\nparakeet\r\n' "$boundary"
    printf -- '--%s\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\nContent-Type: audio/wav\r\n\r\n' "$boundary"
    cat /audio.wav
    printf '\r\n--%s--\r\n' "$boundary"
} > request.body
http POST /v1/audio/transcriptions request.body "multipart/form-data; boundary=$boundary" > response.txt
tr -d '\r' < response.txt | sed -n '1p;$p'
echo
head -n 1 response.txt | grep -Eq '^HTTP/1\.[01] 200 ' || { cat server.log >&2; echo 'Transcription failed' >&2; exit 1; }
# Compare the JSON-escaped expected transcript with the response's text field.
escaped=${expected//\\/\\\\}
escaped=${escaped//\"/\\\"}
grep -Fq "\"text\":\"$escaped\"" response.txt || {
    echo "Transcript mismatch; expected: $expected" >&2
    exit 1
}
kill -0 "$server" || { cat server.log >&2; echo 'Server died during transcription' >&2; exit 1; }
if grep -F '[sched-dbg]' server.log; then
    echo 'The CUDA backend rejected graph nodes (#184)' >&2
    exit 1
fi
echo "inference: exact transcript on the CUDA device"
CHECK
