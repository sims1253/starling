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
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
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
for path in "$cpu" ${vulkan:+"$vulkan"}; do
  if [ ! -f "$path" ]; then
    echo "error: engine binary not found: $path" >&2
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

# Checks one staged engine: --version must report the server/version
# block, --abi-version must equal the header's ABI.
check_engine() {
  local path="$1" backend="$2"
  local version_line abi
  if ! version_line="$("$path" --version 2>/dev/null | head -n 1)"; then
    echo "error: $backend engine failed --version: $path" >&2
    exit 1
  fi
  if [ "${version_line#starling-serve }" = "$version_line" ]; then
    echo "error: $backend engine --version output unrecognized: '$version_line'" >&2
    exit 1
  fi
  abi="$("$path" --abi-version 2>/dev/null | tr -d '[:space:]')"
  if [ "$abi" != "$expected_abi" ]; then
    echo "error: $backend engine speaks ABI $abi, but the app expects $expected_abi; refusing to stage it" >&2
    exit 1
  fi
  echo "$backend: $version_line (abi $abi)"
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

mkdir -p "$out"
rm -f "$out/SHA256SUMS.txt" "$out/engines.json"

# Stage and verify; sums accumulate in preference order (vulkan first,
# matching engines.json and the app's selection order).
sums=""
entries=""
stage_one() {
  local backend="$1" source="$2" name sha
  check_engine "$source" "$backend"
  name="$(layout_name "$backend" "$source")"
  cp "$source" "$out/$name"
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

version="$("$cpu" --version 2>/dev/null | head -n 1 | awk '{print $2}')"
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
