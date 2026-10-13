#!/bin/bash
# Metadata culprit, encoder artifact from the raw export, fd-load diagnosis, LLM prefill scaling, energy.
set -u
export ANDROID_SERIAL=$(adb devices | awk "/^192\.168\.178\.59:[0-9]+\s+device/{print \$1; exit}")
echo "serial: $ANDROID_SERIAL"; [ -n "$ANDROID_SERIAL" ] || { echo "ERROR: phone not on adb"; exit 1; }
LOG=${1:-/tmp/npu-spike/logs/own5}; mkdir -p "$LOG"
REPO=/home/m0hawk/Documents/starling/.claude/worktrees/agent-a250a02653568f706
source $REPO/benchmarks/fast_engine/phone_common.sh
D=/data/local/tmp/npu-spike
O=$D/own
H=/tmp/npu-spike/out/own
state=$(wakefulness); echo "display: $state"
if [ "$state" = "mWakefulness=Awake" ]; then
  fg=$(adb shell "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'" | tr -d '\r')
  case "$fg" in *Launcher*|*launcher*|*SystemUI*|*keyguard*|"") ;; *) echo "PHONE IN USE: $fg"; exit 3 ;; esac
fi
trap 'adb shell "pkill npu_tool; pkill -f pwsample" >/dev/null 2>&1; wake_release || true' EXIT
wake_hold 3600 || exit 1
screen_off || exit 1
adb push -q /tmp/npu-spike/build-npu/npu_tool $O/ >/dev/null
adb shell "rm -rf $O/v $O/*.stnpu* $O/enc_out0.f32; mkdir -p $O/m; df -h /data | tail -1"
for f in gelu_nokeep gelu_nominrt enc_fp16w_nokeep qwen3_prefill_L4_signometa qwen3_prefill_L14_nokeep; do adb push -q $H/$f.tflite $O/m/ >/dev/null; done
IN="$D/mel_w800.f32 $D/kbias_w800.f32"
T="cd $O && ./npu_tool"
step() { echo "===== $1"; shift; timeout 900 adb shell "$*" 2>&1 | grep -v "SouthBound context" | tee -a "$LOG/log.txt"; }
KV9=458752; for i in $(seq 2 9); do KV9="$KV9,458752"; done
KV29=458752; for i in $(seq 2 29); do KV29="$KV29,458752"; done

step "compile gelu, keep_stablehlo_constant renamed" "$T compile $O/m/gelu_nokeep.tflite $O/m/g1.stnpu --fn serving_default --in 1490944 --out 1490944"
step "compile gelu, min_runtime_version renamed" "$T compile $O/m/gelu_nominrt.tflite $O/m/g2.stnpu --fn serving_default --in 1490944 --out 1490944"
step "compile encoder (raw export, keep_stablehlo_constant renamed)" "$T compile $O/m/enc_fp16w_nokeep.tflite $O/enc.stnpu --fn serving_default --in 409600,416 --out 425984"
for i in 1 2 3; do step "encoder load+run, fresh process $i" "$T run $O/enc.stnpu --mode mmap --runs 20 --out $O/enc $IN"; done
adb pull -q $O/enc_out0.f32 "$LOG/enc_out0.f32" >/dev/null
step "repack aligned" "$T repack $O/enc.stnpu $O/al.stnpu --pad 0"
step "fd+offset, firmware-default SQ loading" "STARLING_NPU_SQ_LOAD=default $T run $O/al.stnpu --mode fdoff --runs 5 $IN"
step "fd, firmware-default SQ loading" "STARLING_NPU_SQ_LOAD=default $T run $O/al.stnpu --mode fd --runs 5 $IN"
step "fd+offset, SQ loading mmap" "STARLING_NPU_SQ_LOAD=mmap $T run $O/al.stnpu --mode fdoff --runs 5 $IN"
adb shell "rm -f $O/al.stnpu $O/al.stnpu.bc"
step "compile prefill 4 layers" "$T compile $O/m/qwen3_prefill_L4_signometa.tflite $O/m/p4.stnpu --fn subgraph_0_fn --in 458752 --out $KV9"
step "run prefill 4 layers" "$T run $O/m/p4.stnpu --runs 20"
step "compile prefill 14 layers" "$T compile $O/m/qwen3_prefill_L14_nokeep.tflite $O/m/p14.stnpu --fn serving_default --in 458752 --out $KV29"
step "run prefill 14 layers" "$T run $O/m/p14.stnpu --runs 20"
adb logcat -d -v brief | grep -iE "Darwinn TFlite ODC failed" | cut -c1-300 | tail -6
adb shell "rm -rf $O/m"

adb shell "dumpsys battery | grep -E '^  status|AC powered|USB powered'" | tr -d '\r' | tee "$LOG/battery.txt"
if grep -qE "status: 3$" "$LOG/battery.txt"; then
  adb shell 'cat > /data/local/tmp/pwsample.sh' <<'S'
B=/sys/class/power_supply/battery
while true; do read u _ < /proc/uptime; echo "$u $(cat $B/current_now) $(cat $B/voltage_now)"; sleep 0.2; done
S
  window() { local tag=$1; shift
    adb shell "sh /data/local/tmp/pwsample.sh > $O/pw_$tag.txt & echo \$! > $O/pw.pid; $*; kill \$(cat $O/pw.pid)" > "$LOG/$tag.log" 2>&1
    adb pull -q $O/pw_$tag.txt "$LOG/" >/dev/null; }
  for rep in 1 2; do
    sleep 20; window idle$rep "sleep 40"
    sleep 10; window npu100_$rep "$T run $O/enc.stnpu --mode mmap --runs 100 $IN"
    sleep 10; window npu2100_$rep "$T run $O/enc.stnpu --mode mmap --runs 2100 $IN"
  done
else
  echo "NOT ON BATTERY: energy skipped"
fi
echo done
