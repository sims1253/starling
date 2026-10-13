#!/usr/bin/env bash
# npu_compile_all.sh — one command: compile the TPU audio encoder of every
# Qwen3-ASR model the installed Starling app has loaded, on a Pixel (Tensor G5),
# and install it into the app.
#
#   scripts/npu/npu_compile_all.sh [--gguf HOST_GGUF ...] [--models DIR ...]
#                                  [--all] [--app PKG] [--wait SECONDS] [--dry-run]
#
# WARNING: the phone's on-device compiler can exhaust the phone's memory and
# panic its kernel (the phone reboots). Only the Qwen3-ASR encoder, a graph
# with a clean record, is built here, and npu_compile_phone.sh refuses to
# compile with less than 4 GB available; still, save your work on the phone
# first. See docs/fast-engine.md, "Pixel TPU encoder".
#
# How it knows what to build: when the app loads a model, the engine writes
# status-<piece>-<model hash>.txt next to the artifacts it looks for (files/npu
# of the app, and its external files dir), with the model hash (a digest of
# the GGUF's encoder tensors and metadata, naming the artifact), the GGUF file it loaded
# and state missing / stale (an artifact compiled for older TPU firmware or
# by an older exporter) / ok / error. So: load each model in the app once,
# then run this. It
#   1. finds the phone: $ANDROID_SERIAL, else the single `adb devices` entry,
#      else wireless debugging via mDNS (`adb mdns services`, avahi-browse)
#      and `adb connect`; with --wait it keeps looking that long;
#   2. finds the installed Starling packages (dev.starling.mobile and its
#      .debug / .experimental variants; --app picks one);
#   3. reads their status files (run-as for a debuggable package, the
#      external files dir otherwise) and, for every missing or stale
#      Qwen3-ASR encoder (--all: also the ok ones), finds the GGUF with that
#      model hash on this host and compiles it with npu_compile.sh into every
#      package that asked for it.
# The graph is exported from the very GGUF the app loads (its own encoder
# weights), so the host needs that file, found by its model hash:
#   * --gguf HOST_GGUF (repeatable): that file, if its hash is one the app
#     asked for;
#   * else the first GGUF with the hash under --models DIR (repeatable;
#     default $STARLING_MODELS_DIR, else the repository's models/ directory,
#     else that of the main checkout when this is a git worktree). Hashes are
#     cached by path, size and mtime in ~/.cache/starling-npu;
#   * else the file the status names, pulled from the phone (run-as for a
#     debuggable package; otherwise only if the adb shell can read it);
#   * else that model is refused: pass --gguf with the file the app loads.
# Nothing is precompiled or shipped: the phone's own compiler builds every
# artifact for its firmware, and after a firmware update the app reports the
# files as stale until this is rerun.
#
# Env: PY (python with gguf, litert-torch, ai-edge-quantizer, torch; xxhash
# speeds up hashing), NPU_TOOL and STARLING_PHONE_LOCK_HELD (see
# npu_compile.sh).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python3}
ALL=0; ONLY=""; WAIT=0; DRY=0; GGUFS=(); MODELS=()
while [ $# -gt 0 ]; do
  case $1 in
    --gguf) GGUFS+=("$2"); shift ;;
    --models) MODELS+=("$2"); shift ;;
    --all) ALL=1 ;;
    --app) ONLY=$2; shift ;;
    --wait) WAIT=$2; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n 2,49p "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
for g in "${GGUFS[@]}"; do [ -f "$g" ] || { echo "--gguf $g: no such file" >&2; exit 2; }; done
if [ ${#MODELS[@]} = 0 ]; then
  if [ -n "${STARLING_MODELS_DIR:-}" ]; then MODELS=("$STARLING_MODELS_DIR")
  elif [ -d "$ROOT/models" ]; then MODELS=("$ROOT/models")
  elif common=$(git -C "$ROOT" rev-parse --path-format=absolute --git-common-dir 2>/dev/null) &&
       [ -d "$(dirname "$common")/models" ]; then MODELS=("$(dirname "$common")/models")
  fi
fi
EXPORTER=$ROOT/scripts/npu/export_encoder.py
# One argument quoted for the phone's shell.
q() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
same_hash() { "$PY" -I -c 'import sys; print(int(sys.argv[1], 16) == int(sys.argv[2], 16))' "$1" "$2" 2>/dev/null | grep -qx True; }

# ---- 1. the phone ----
online() { [ "$(adb -s "$1" get-state 2>/dev/null)" = device ]; }
find_phone() {
  if [ -n "${ANDROID_SERIAL:-}" ] && online "$ANDROID_SERIAL"; then echo "$ANDROID_SERIAL"; return 0; fi
  local devs
  devs=$(adb devices | awk 'NR > 1 && $2 == "device" {print $1}')
  if [ "$(echo "$devs" | grep -c .)" = 1 ]; then echo "$devs"; return 0; fi
  if [ -n "$devs" ]; then echo "several phones attached; set ANDROID_SERIAL" >&2; return 2; fi
  local s
  for s in $( (adb mdns services 2>/dev/null | awk '/_adb-tls-connect/ {print $NF}'
              command -v avahi-browse >/dev/null &&
                timeout 10 avahi-browse -rpt _adb-tls-connect._tcp 2>/dev/null |
                awk -F';' '$1 == "=" && $3 == "IPv4" {print $8 ":" $9}') | sort -u); do
    timeout 10 adb connect "$s" >/dev/null 2>&1 || true
    if online "$s"; then echo "$s"; return 0; fi
  done
  return 1
}
deadline=$((SECONDS + WAIT))
until SERIAL=$(find_phone); do
  [ $? = 2 ] && exit 2
  if [ $SECONDS -ge $deadline ]; then
    echo "no phone found (USB, or wireless debugging; pair it with 'adb pair' first)" >&2
    exit 1
  fi
  sleep 15
done
export ANDROID_SERIAL=$SERIAL
echo "phone: $SERIAL ($(adb shell getprop ro.product.model </dev/null | tr -d '\r'))"

# ---- 2. the app ----
PKGS=$(adb shell pm list packages dev.starling.mobile </dev/null | tr -d '\r' | sed -n 's/^package://p' |
       grep -E '^dev\.starling\.mobile(\.debug|\.experimental)?$' || true)
[ -n "$ONLY" ] && PKGS=$(echo "$PKGS" | grep -Fx "$ONLY" || true)
[ -n "$PKGS" ] || { echo "no Starling app installed${ONLY:+ ($ONLY)}" >&2; exit 1; }

# ---- 3. what the app asked for ----
WORK=$(mktemp -d /tmp/npu-compile-all.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
STATUS=$WORK/status
for p in $PKGS; do
  if adb shell "run-as $p true" </dev/null >/dev/null 2>&1; then
    how="debuggable: files/npu via run-as"
    adb shell "run-as $p sh -c 'cat files/npu/status-*.txt 2>/dev/null'" </dev/null | tr -d '\r' | sed "s/^piece=/pkg=$p\npiece=/" >>"$STATUS" || true
  else
    how="release: external files dir"
  fi
  adb shell "cat /sdcard/Android/data/$p/files/npu/status-*.txt 2>/dev/null" </dev/null | tr -d '\r' | sed "s/^piece=/pkg=$p\npiece=/" >>"$STATUS" || true
  echo "app: $p ($how)"
done
touch "$STATUS"
# One tab-separated line per encoder model hash:
#   piece hash hd states packages [package gguf-on-the-phone]...
PLAN=$("$PY" -I - "$STATUS" "$ALL" <<'EOF'
import sys
recs, cur = [], None
for line in open(sys.argv[1]):
    k, _, v = line.rstrip("\n").partition("=")
    if k == "pkg":
        cur = {"pkg": v}
        recs.append(cur)
    elif cur is not None:
        cur[k] = v
want = {}
for r in recs:
    if r.get("piece") != "qwen3-asr-enc" or not r.get("hash"):
        continue                                  # the only piece built here
    m = dict(kv.split("=", 1) for kv in r.get("model", "").split() if "=" in kv)
    state = r.get("state", "")
    if state not in ("missing", "stale", "error") and not (sys.argv[2] == "1" and state == "ok"):
        continue
    w = want.setdefault((r["piece"], r["hash"].lower()), {"hd": m.get("hd", ""), "pkgs": set(), "states": set(),
                                                          "src": {}})
    w["pkgs"].add(r["pkg"])
    w["states"].add(state)
    if r.get("gguf") and "\t" not in r["gguf"]:
        w["src"].setdefault(r["pkg"], r["gguf"])
for (piece, h), w in sorted(want.items()):
    src = [x for p, g in sorted(w["src"].items()) for x in (p, g)]
    print("\t".join([piece, h, w["hd"] or "-", "/".join(sorted(w["states"])), ",".join(sorted(w["pkgs"]))] + src))
EOF
)
if [ -z "$PLAN" ]; then
  echo "nothing to compile: every Qwen3-ASR TPU encoder the app asked for is up to date."
  echo "(Load a Qwen3-ASR model in the app once so it reports what it needs, then rerun.)"
  exit 0
fi
echo "to compile:"; echo "$PLAN" | cut -f1-5 | tr '\t' ' ' | sed 's/^/  /'
[ $DRY = 1 ] && exit 0

# ---- 4. the GGUF of each, on this host ----
GIVEN=()   # "hash<TAB>path" of each --gguf
for g in "${GGUFS[@]}"; do
  h=$("$PY" -I "$EXPORTER" --model-hash "$g" | awk 'NR == 1 {print $1}') || { echo "!! --gguf $g: not a Qwen3-ASR GGUF" >&2; exit 2; }
  GIVEN+=("$h"$'\t'"$g")
done
USED=()   # the GGUFs compiled
PHONE_LOCK=(flock /tmp/starling-phone.lock)
[ -n "${STARLING_PHONE_LOCK_HELD:-}" ] && PHONE_LOCK=()   # the caller holds it
# pull HASH PKG PATH: the GGUF the app loaded, copied from the phone; prints
# the local copy if its model hash is HASH.
pull() {
  local dest=$WORK/pulled-$1.gguf
  if adb shell "run-as $2 true" </dev/null >/dev/null 2>&1; then
    "${PHONE_LOCK[@]}" adb exec-out "run-as $2 cat $(q "$3")" </dev/null >"$dest" 2>/dev/null || true
  else
    "${PHONE_LOCK[@]}" adb exec-out "cat $(q "$3")" </dev/null >"$dest" 2>/dev/null || true
  fi
  if [ -s "$dest" ] && same_hash "$1" "$("$PY" -I "$EXPORTER" --model-hash "$dest" 2>/dev/null | awk '{print $1}')"; then
    echo "$dest"
  else
    rm -f "$dest"
    return 1
  fi
}
# gguf_for HASH HD [PKG PATH]...: a host GGUF whose model hash is HASH.
gguf_for() {
  local want=$1 hd=$2 e f
  shift 2
  for e in "${GIVEN[@]}"; do
    if same_hash "$want" "${e%%$'\t'*}"; then echo "${e#*$'\t'}"; return 0; fi
  done
  if [ ${#MODELS[@]} -gt 0 ]; then
    local hdarg=()
    [ "$hd" != - ] && hdarg=(--hd "$hd")
    if f=$("$PY" -I "$EXPORTER" --find "$want" "${hdarg[@]}" "${MODELS[@]}"); then echo "$f"; return 0; fi
  fi
  while [ $# -ge 2 ]; do
    echo "   not on this host; pulling $2 from $1" >&2
    if f=$(pull "$want" "$1" "$2"); then echo "$f"; return 0; fi
    echo "   could not pull it (or it has another model hash)" >&2
    shift 2
  done
  return 1
}

fail=0
while IFS=$'\t' read -r -a F; do
  piece=${F[0]} hash=${F[1]} hd=${F[2]} states=${F[3]} pkgs=${F[4]}
  echo "== $piece $hash ($states) -> $pkgs"
  if ! gguf=$(gguf_for "$hash" "$hd" "${F[@]:5}"); then
    echo "!! $piece $hash: no GGUF with this model hash on this host${MODELS[*]:+ (searched ${MODELS[*]})}" >&2
    for ((i = 5; i + 1 < ${#F[@]}; i += 2)); do echo "   ${F[i]} loads ${F[i+1]}" >&2; done
    echo "   pass --gguf HOST_GGUF with a copy of the file the app loads" >&2
    fail=1; continue
  fi
  echo "   GGUF: $gguf"
  USED+=("$gguf")
  PY=$PY "$ROOT/scripts/npu/npu_compile.sh" "$gguf" --hash "$hash" --app "$pkgs" </dev/null ||
    { echo "!! $piece $hash failed" >&2; fail=1; }
done <<<"$PLAN"
for e in "${GIVEN[@]}"; do
  g=${e#*$'\t'}
  printf '%s\n' "${USED[@]}" | grep -qxF "$g" ||
    { echo "!! --gguf $g (model hash ${e%%$'\t'*}) is not a model the app asked for" >&2; fail=1; }
done
if [ $fail = 0 ]; then echo "done. Restart the app (or reload the model) to pick the TPU files up."
else echo "done, with failures (!! above). Restart the app (or reload the model) to pick up what was built."
fi
exit $fail
