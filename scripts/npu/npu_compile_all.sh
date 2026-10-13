#!/usr/bin/env bash
# npu_compile_all.sh — one command: compile the TPU audio encoder of every
# Qwen3-ASR model the installed Starling app has loaded, on a Pixel (Tensor G5),
# and install it into the app.
#
#   scripts/npu/npu_compile_all.sh [--all] [--app PKG] [--wait SECONDS] [--dry-run]
#
# WARNING: the phone's on-device compiler can exhaust the phone's memory and
# panic its kernel (the phone reboots). Only the Qwen3-ASR encoder, a graph
# with a clean record, is built here, and npu_compile_phone.sh refuses to
# compile with less than 4 GB available; still, save your work on the phone
# first. See docs/fast-engine.md, "Pixel TPU encoder".
#
# How it knows what to build: when the app loads a model, the engine writes
# status-<piece>-<fingerprint>.txt next to the artifacts it looks for (files/npu
# of the app, and its external files dir), with state missing / stale (an
# artifact compiled for older TPU firmware or by an older exporter) / ok /
# error. So: load each model in the app once, then run this. It
#   1. finds the phone: $ANDROID_SERIAL, else the single `adb devices` entry,
#      else wireless debugging via mDNS (`adb mdns services`, avahi-browse)
#      and `adb connect`; with --wait it keeps looking that long;
#   2. finds the installed Starling packages (dev.starling.mobile and its
#      .debug / .experimental variants; --app picks one);
#   3. reads their status files (run-as for a debuggable package, the
#      external files dir otherwise) and compiles every missing or stale
#      Qwen3-ASR encoder (--all: also the ok ones) with npu_compile.sh, each
#      into every package that asked for it.
# The HF checkpoint is downloaded on first use (Qwen/Qwen3-ASR-0.6B-hf,
# Qwen/Qwen3-ASR-1.7B-hf; weights and config only). Nothing is precompiled or
# shipped: the phone's own compiler builds every artifact for its firmware,
# and after a firmware update the app reports the files as stale until this
# is rerun.
#
# Env: PY (python with litert-torch, ai-edge-quantizer, transformers, torch,
# huggingface_hub), NPU_TOOL (see npu_compile.sh).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python3}
ALL=0; ONLY=""; WAIT=0; DRY=0
while [ $# -gt 0 ]; do
  case $1 in
    --all) ALL=1 ;;
    --app) ONLY=$2; shift ;;
    --wait) WAIT=$2; shift ;;
    --dry-run) DRY=1 ;;
    -h|--help) sed -n 2,35p "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done

# (piece, arch, LLM hidden size) -> HF checkpoint. Nothing else is compiled.
checkpoint() {
  case "$1:$2:$3" in
    qwen3-asr-enc:qwen3:1024) echo Qwen/Qwen3-ASR-0.6B-hf ;;
    qwen3-asr-enc:qwen3:2048) echo Qwen/Qwen3-ASR-1.7B-hf ;;
    *) echo "" ;;
  esac
}

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
STATUS=$(mktemp /tmp/npu-status.XXXXXX)
trap 'rm -f "$STATUS"' EXIT
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
# One line per encoder and fingerprint: piece hash arch hd state packages
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
    if r.get("piece") != "qwen3-asr-enc":
        continue                                  # the only piece built here
    m = dict(kv.split("=", 1) for kv in r.get("model", "").split() if "=" in kv)
    state = r.get("state", "")
    if state not in ("missing", "stale", "error") and not (sys.argv[2] == "1" and state == "ok"):
        continue
    key = (r["piece"], r["hash"], m.get("arch", "?"), m.get("hd", "?"))
    want.setdefault(key, [set(), set()])
    want[key][0].add(r["pkg"])
    want[key][1].add(state)
for (piece, h, arch, hd), (pkgs, states) in sorted(want.items()):
    print(piece, h, arch, hd, "/".join(sorted(states)), ",".join(sorted(pkgs)))
EOF
)
if [ -z "$PLAN" ]; then
  echo "nothing to compile: every Qwen3-ASR TPU encoder the app asked for is up to date."
  echo "(Load a Qwen3-ASR model in the app once so it reports what it needs, then rerun.)"
  exit 0
fi
echo "to compile:"; echo "$PLAN" | sed 's/^/  /'
[ $DRY = 1 ] && exit 0

fail=0
while read -r piece hash arch hd states pkgs; do
  ck=$(checkpoint "$piece" "$arch" "$hd")
  if [ -z "$ck" ]; then echo "!! no checkpoint known for $piece ($arch, hd=$hd); skipped" >&2; fail=1; continue; fi
  echo "== $piece $hash ($ck, $states) -> $pkgs"
  "$PY" -I -c "import sys; from huggingface_hub import snapshot_download as d; d(sys.argv[1], allow_patterns=['*.json', '*.safetensors'])" "$ck" >/dev/null
  PY=$PY "$ROOT/scripts/npu/npu_compile.sh" "$ck" --hash "$hash" --app "$pkgs" </dev/null || { echo "!! $piece $hash failed" >&2; fail=1; }
done <<<"$PLAN"
echo "done. Restart the app (or reload the model) to pick the TPU files up."
exit $fail
