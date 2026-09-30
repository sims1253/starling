#!/usr/bin/env bash
# Stage a development engines directory for the desktop app (#362).
#
# Usage:
#   scripts/stage-engines.sh --cpu <path> [--vulkan <path>] --out <dir>
#
# Copies the given server binaries under the bundled layout names
# (starling-serve-cpu[-vulkan][.exe]), checks each with --version and
# --abi-version (refusing to stage an engine that does not answer), and
# writes engines.json + SHA256SUMS.txt in the layout the app's engine
# discovery expects.
#
# Then run the app with:
#   STARLING_ENGINE_DIR=<dir> cargo run -p app
#
# The bundled layout (see crates/dictation/src/engine/bundle.rs):
#   <dir>/starling-serve-cpu[.exe]
#   <dir>/starling-serve-vulkan[.exe]
#   <dir>/SHA256SUMS.txt
#   <dir>/engines.json

set -euo pipefail

usage() {
  cat <<'EOF'
Stage a development engines directory for the desktop app (#362).

Usage:
  scripts/stage-engines.sh --cpu <path> [--vulkan <path>] --out <dir>

Copies the given server binaries under the bundled layout names
(starling-serve-cpu[-vulkan][.exe]), checks each with --version and
--abi-version (refusing to stage an engine that does not answer), and
writes engines.json + SHA256SUMS.txt in the layout the app's engine
discovery expects.

Then run the app with:
  STARLING_ENGINE_DIR=<dir> cargo run -p app

The bundled layout (see crates/dictation/src/engine/bundle.rs):
  <dir>/starling-serve-cpu[.exe]
  <dir>/starling-serve-vulkan[.exe]
  <dir>/SHA256SUMS.txt
  <dir>/engines.json
EOF
}

# The ABI this app expects; must equal STARLING_GGML_ABI_VERSION in
# cpp/include/starling_ggml.h (the lockstep test in
# crates/dictation/src/engine/mod.rs pins it).
expected_abi="$(sed -n 's/^#define STARLING_GGML_ABI_VERSION[[:space:]]*\([0-9]\+\).*/\1/p' \
  "$(dirname "$0")/../../../cpp/include/starling_ggml.h")"
if [ -z "$expected_abi" ]; then
  echo "error: cannot read STARLING_GGML_ABI_VERSION from cpp/include/starling_ggml.h" >&2
  exit 1
fi

cpu=""
vulkan=""
out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cpu) cpu="${2:?--cpu needs a path}"; shift 2 ;;
    --vulkan) vulkan="${2:?--vulkan needs a path}"; shift 2 ;;
    --out) out="${2:?--out needs a directory}"; shift 2 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ -z "$cpu" ] || [ -z "$out" ]; then
  echo "error: --cpu <path> and --out <dir> are required (--vulkan optional)" >&2
  exit 1
fi
# Every engine binary to stage, as an array so paths with spaces stay
# one word each.
engines=("$cpu")
if [ -n "$vulkan" ]; then
  engines+=("$vulkan")
fi
for path in "${engines[@]}"; do
  if [ ! -f "$path" ]; then
    echo "error: engine binary not found: $path" >&2
    exit 1
  fi
  if [ ! -x "$path" ]; then
    echo "error: engine binary not executable: $path" >&2
    exit 1
  fi
done

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    echo "error: need sha256sum or shasum to write SHA256SUMS.txt" >&2
    exit 1
  fi
}

# Checks one engine: --version must report the server/version block,
# --abi-version must equal the header's ABI. Prints ONLY the validated
# version on stdout (errors go to stderr, failures exit non-zero), so a
# caller captures it via command substitution:
#   if ! version="$(check_engine ...)"; then exit 1; fi
check_engine() {
  local path="$1" backend="$2"
  local version_line abi_output abi version
  if ! version_line="$("$path" --version 2>/dev/null | head -n 1)"; then
    echo "error: $backend engine failed --version: $path" >&2
    exit 1
  fi
  if [[ ! "$version_line" =~ ^starling-serve[[:space:]]+[^[:space:]]+ ]]; then
    echo "error: $backend engine --version output unrecognized: '$version_line' (from $path)" >&2
    exit 1
  fi
  if ! abi_output="$("$path" --abi-version 2>&1)"; then
    echo "error: $backend engine failed --abi-version: $abi_output" >&2
    exit 1
  fi
  abi="$(printf '%s' "$abi_output" | tr -d '[:space:]')"
  if [ "$abi" != "$expected_abi" ]; then
    echo "error: $backend engine speaks ABI $abi, but the app expects $expected_abi; refusing to stage it" >&2
    exit 1
  fi
  # The version is the second whitespace-separated field of the validated
  # line, whatever spacing separates the fields.
  read -r _ version _ <<<"$version_line"
  printf '%s\n' "$version"
}

# The staged file name keeps a Windows .exe suffix when the source has one.
layout_name() {
  local backend="$1" source="$2"
  local name="starling-serve-$backend"
  case "$source" in
    *.exe) echo "$name.exe" ;;
    *) echo "$name" ;;
  esac
}

# The absolute path of a file (its directory resolved), for the
# same-file guard below: cp refuses to copy a file onto itself.
abs_path() {
  printf '%s/%s\n' "$(cd -- "$(dirname -- "$1")" && pwd -P)" "$(basename -- "$1")"
}

mkdir -p "$out"
# Remove the manifests and any stale layout name NOT staged in this run:
# a vulkan engine staged once, then only cpu again must not linger as a
# selectable engine with no checksum entry. The names this run stages are
# left alone — the source may be that very file (re-staging from the
# output dir).
cpu_name="$(layout_name cpu "$cpu")"
vulkan_name=""
if [ -n "$vulkan" ]; then
  vulkan_name="$(layout_name vulkan "$vulkan")"
fi

# Validate everything BEFORE touching the staged directory (#366): each
# engine's --version/--abi-version, and the cpu/vulkan version agreement,
# run before any manifest is removed or file copied — a validation failure
# must neither leave a half-staged dir (binaries + sums but no engines.json)
# nor destroy a previous good staging.
cpu_version=""
vulkan_version=""
if [ -n "$vulkan" ]; then
  if ! vulkan_version="$(check_engine "$vulkan" vulkan)"; then
    exit 1
  fi
fi
if ! cpu_version="$(check_engine "$cpu" cpu)"; then
  exit 1
fi
if [ -z "$cpu_version" ]; then
  echo "error: could not determine the engine version from --version output" >&2
  exit 1
fi
if [ -n "$vulkan_version" ] && [ "$vulkan_version" != "$cpu_version" ]; then
  echo "error: staged engines disagree on version: cpu reports $cpu_version, vulkan reports $vulkan_version; refusing to stage" >&2
  exit 1
fi

# A failure from here on (copy or hashing I/O) must not leave the dir
# half-staged either: remove the manifests and every layout file this run
# copied (never a source that was re-staged from the output dir itself —
# the same-file guard below skips, not copies, those).
staged_files=()
unstage_on_failure() {
  local status=$?
  if [ "$status" -ne 0 ]; then
    rm -f "$out/SHA256SUMS.txt" "$out/engines.json"
    if [ "${#staged_files[@]}" -gt 0 ]; then
      rm -f "${staged_files[@]}"
    fi
  fi
}
trap unstage_on_failure EXIT

rm -f "$out/SHA256SUMS.txt" "$out/engines.json"
for stale in starling-serve-cpu starling-serve-cpu.exe \
             starling-serve-vulkan starling-serve-vulkan.exe; do
  if [ "$stale" != "$cpu_name" ] && [ "$stale" != "$vulkan_name" ]; then
    rm -f "$out/$stale"
  fi
done

# Stage and verify; sums accumulate in preference order (vulkan first,
# matching engines.json and the app's selection order).
sums=""
entries=""
stage_one() {
  local backend="$1" source="$2" name sha
  case "$backend" in
    cpu) echo "cpu: starling-serve $cpu_version (abi $expected_abi)" ;;
    vulkan) echo "vulkan: starling-serve $vulkan_version (abi $expected_abi)" ;;
  esac
  name="$(layout_name "$backend" "$source")"
  # Re-staging from the output dir must be idempotent: skip the copy when
  # source and destination are the same file.
  if [ "$(abs_path "$source")" != "$(abs_path "$out/$name")" ]; then
    cp "$source" "$out/$name"
    staged_files+=("$out/$name")
  fi
  chmod +x "$out/$name"
  sha="$(sha256_of "$out/$name")"
  sums="$sums$sha  $name
"
  entries="$entries{\"backend\":\"$backend\",\"file\":\"$name\"},"
}

if [ -n "$vulkan" ]; then
  stage_one vulkan "$vulkan"
fi
stage_one cpu "$cpu"
entries="${entries%,}"

printf '%s' "$sums" > "$out/SHA256SUMS.txt"

# engines.json carries one version for the whole bundle: the validated
# --version lines agreed above, so this is the version both engines report.
version="$cpu_version"
cat > "$out/engines.json" <<EOF
{
  "version": "$version",
  "abi": $expected_abi,
  "engines": [
    $entries
  ]
}
EOF

echo "staged engines in $out:"
ls -l "$out"
echo
echo "Run the app with: STARLING_ENGINE_DIR=$out"
