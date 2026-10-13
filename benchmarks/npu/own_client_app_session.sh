#!/bin/bash
# App-side execution with a shell-compiled artifact, EdgeTPU allowlist probe,
# LLM-prefill / Parakeet compile probes.
set -u
export ANDROID_SERIAL=$(adb devices | awk "/^192\.168\.178\.59:[0-9]+\s+device/{print \$1; exit}")
echo "serial: $ANDROID_SERIAL"; [ -n "$ANDROID_SERIAL" ] || { echo "ERROR: phone not on adb"; exit 1; }
LOG=${1:-/tmp/npu-spike/logs/own4}; mkdir -p "$LOG"
REPO=/home/m0hawk/Documents/starling/.claude/worktrees/agent-a250a02653568f706
source $REPO/benchmarks/fast_engine/phone_common.sh
D=/data/local/tmp/npu-spike
O=$D/own
A=$O/v/qwen3_enc_w800_fp16wmask_signometa.stnpu
PKG=dev.starling.npuprobe
EXT=/sdcard/Android/data/$PKG/files
state=$(wakefulness); echo "display: $state"
if [ "$state" = "mWakefulness=Awake" ]; then
  fg=$(adb shell "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'" | tr -d '\r')
  case "$fg" in *Launcher*|*launcher*|*SystemUI*|*keyguard*|"") ;; *) echo "PHONE IN USE: $fg"; exit 3 ;; esac
fi
trap 'adb shell "pkill npu_tool" >/dev/null 2>&1; wake_release || true' EXIT
wake_hold 3600 || exit 1
screen_off || exit 1
step() { echo "===== $1"; shift; timeout 900 adb shell "$*" 2>&1 | grep -v "SouthBound context" | tee -a "$LOG/log.txt"; }

echo "===== allowlist probe"
adb shell 'ls -la /vendor/etc | grep -iE "edgetpu|tpu|darwinn|allow"; for f in $(ls /vendor/etc 2>/dev/null | grep -iE "edgetpu|allow"); do echo "-- $f"; head -c 3000 /vendor/etc/$f 2>&1; echo; done' 2>&1 | tee "$LOG/allowlist.txt" | head -60
adb shell 'device_config list 2>/dev/null | grep -iE "edgetpu|tpu|darwinn" | head; getprop | grep -iE "edgetpu|tpu|darwinn" | head -20' 2>&1 | tee -a "$LOG/allowlist.txt"

echo "===== app: execute a shell-compiled artifact"
adb install -r -t /tmp/npu-spike/build-npu/apk/npuprobe.apk 2>&1 | tail -1
adb shell "cp $A $EXT/encoder.stnpu && ls -la $EXT"
adb logcat -c
for i in 1 2; do step "app: run (ext artifact) $i" "am instrument -w -e step run -e art ext -e runs 20 $PKG/.ProbeInstrumentation"; done
adb logcat -d -v brief | grep -iE "StarlingNpu|edgetpu|allow|tachyon|darwinn|avc" | grep -v settaskprofile | cut -c1-300 | tail -40 > "$LOG/app.logcat"
grep -iE "allow|denied|error" "$LOG/app.logcat" | head -8
adb pull -q $EXT/app_out0.f32 "$LOG/app_out0.f32" >/dev/null 2>&1

echo "===== LLM prefill / Parakeet compile probes (shell)"
adb shell "mkdir -p $O/p"
for m in qwen3_prefill_L4 parakeet_subsample parakeet_block; do adb push -q /tmp/npu-spike/out/own/${m}_signometa.tflite $O/p/$m.tflite >/dev/null; done
T="cd $O && ./npu_tool"
step "compile qwen3 prefill (4 layers, S=112)" "$T compile $O/p/qwen3_prefill_L4.tflite $O/p/prefill.stnpu --fn subgraph_0_fn --in 458752 --out 458752,458752,458752,458752,458752,458752,458752,458752,458752"
step "run qwen3 prefill" "$T run $O/p/prefill.stnpu --runs 20"
step "compile parakeet subsampling" "$T compile $O/p/parakeet_subsample.tflite $O/p/sub.stnpu --fn subgraph_0_fn --in 380928 --out 380928"
step "run parakeet subsampling" "$T run $O/p/sub.stnpu --runs 20"
step "compile parakeet conformer block" "$T compile $O/p/parakeet_block.tflite $O/p/blk.stnpu --fn subgraph_0_fn --in 380928,757760 --out 380928"
step "run parakeet conformer block" "$T run $O/p/blk.stnpu --runs 20"
adb logcat -d -v brief | grep -iE "Darwinn TFlite ODC failed|error:" | cut -c1-300 | tail -10
adb shell "rm -rf $O/p"
echo done
