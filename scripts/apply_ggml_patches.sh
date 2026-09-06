#!/usr/bin/env bash
# Apply the ordered ggml patch series without changing the caller's Git index.
# Usage: bash scripts/apply_ggml_patches.sh
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GGML_DIR="${PROJECT_ROOT}/third_party/ggml"
PATCH_DIR="${PROJECT_ROOT}/third_party/ggml-patches"

if [[ ! -d "${GGML_DIR}/.git" && ! -f "${GGML_DIR}/.git" ]]; then
    echo "error: ggml submodule not initialized at ${GGML_DIR}" >&2
    echo "       run git submodule update --init --recursive" >&2
    exit 1
fi
if [[ ! -d "${PATCH_DIR}" ]]; then
    echo "error: patch directory not found at ${PATCH_DIR}" >&2
    exit 1
fi

# Bash glob order under the C locale follows the numeric filename prefixes.
shopt -s nullglob
PATCHES=("${PATCH_DIR}"/*.patch)
if [[ ${#PATCHES[@]} -eq 0 ]]; then
    echo "ggml patches: no patches found (nothing to do)"
    exit 0
fi
cd "${GGML_DIR}"

# mkdir is atomic on Linux, macOS and Git Bash; flock is not always available.
LOCK_DIR="$(git rev-parse --absolute-git-dir)/starling-patches.lock"
for ((attempt = 0; ; attempt++)); do
    if mkdir "${LOCK_DIR}" 2>/dev/null; then
        break
    fi
    if [[ ! -d "${LOCK_DIR}" || ${attempt} -ge 60 ]]; then
        echo "error: cannot acquire ggml patch lock: ${LOCK_DIR}" >&2
        echo "       if no patch process is running, remove that directory and retry" >&2
        exit 1
    fi
    sleep 1
done
TEMP_DIR=""
cleanup() {
    [[ -z "${TEMP_DIR}" ]] || rm -rf "${TEMP_DIR}"
    rmdir "${LOCK_DIR}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
TEMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/starling-ggml-patches.XXXXXX")"
export GIT_INDEX_FILE="${TEMP_DIR}/index"

# Snapshot the worktree, including local edits and untracked files, in a private
# index. Reverse the series backwards so later overlapping hunks are removed
# before checking their predecessors. Nothing touches the worktree yet.
git read-tree HEAD
git add --all -- .
BEFORE="$(git write-tree)"
for ((i = ${#PATCHES[@]} - 1; i >= 0; i--)); do
    if git apply --cached --check --reverse "${PATCHES[i]}" 2>/dev/null; then
        git apply --cached --reverse "${PATCHES[i]}"
    fi
done

# Every patch must now apply, including predecessors independent of the tail.
# A partial hunk or conflicting edit fails before any real file is changed.
for patch in "${PATCHES[@]}"; do
    if ! git apply --cached "${patch}"; then
        echo "error: cannot validate complete ggml patch series at $(basename "${patch}")" >&2
        echo "       worktree and Git index unchanged; inspect git -C '${GGML_DIR}' status" >&2
        exit 1
    fi
done
AFTER="$(git write-tree)"
if [[ "${BEFORE}" == "${AFTER}" ]]; then
    echo "ggml patches: complete series already applied (nothing to do)"
    exit 0
fi

# Apply only the missing changes. git apply checks the entire diff before
# writing, and preserves unrelated local edits and the original staging index.
git diff --binary --no-ext-diff --no-textconv --src-prefix=a/ --dst-prefix=b/ "${BEFORE}" "${AFTER}" > "${TEMP_DIR}/remaining.patch"
git apply --check "${TEMP_DIR}/remaining.patch"
git apply "${TEMP_DIR}/remaining.patch"
echo "ggml patches: complete series applied"
