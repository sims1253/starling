#!/bin/bash
# Tensor G5 NPU spike, phone side: masked Qwen3-ASR encoder windows on the NPU (JIT through
# the vendor compiler in /vendor/lib64/libedgetpu_litert.so), then battery current x voltage
# sampling for NPU vs GPU encoder energy. Energy needs the phone unplugged; the GPU side uses
# a starling-bench built with gpu_encoder_only_timing.patch, pushed to $D/sbenc2.
# Note: the LiteRT 2.3.0 JIT cache (NPU_CACHE_DIR) writes but fails to reload (DISPATCH_OP
# prepare error), so the cached-compile step below is expected to FAIL.
set -u
export ANDROID_SERIAL=$(adb devices | awk "/^192\.168\.178\.59:[0-9]+\s+device/{print \$1; exit}")
echo "serial: $ANDROID_SERIAL"; [ -n "$ANDROID_SERIAL" ] || { echo "ERROR: phone not on adb"; exit 1; }
LOG=${1:-/tmp/npu-spike/logs/s4}; mkdir -p "$LOG"
REPO=/home/m0hawk/Documents/starling/.claude/worktrees/agent-a250a02653568f706
source $REPO/benchmarks/fast_engine/phone_common.sh
D=/data/local/tmp/npu-spike
G=/data/local/tmp/starling/qwen3-asr-0.6b-q4e8.gguf
state=$(wakefulness); echo "display: $state"
if [ "$state" = "mWakefulness=Awake" ]; then
  fg=$(adb shell "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'" | tr -d '\r')
  case "$fg" in *Launcher*|*launcher*|*SystemUI*|*keyguard*|"") ;; *) echo "PHONE IN USE: $fg"; exit 3 ;; esac
fi
stop_sampler() { adb shell "p=\$(cat $D/pwr.pid 2>/dev/null) && kill \$p; rm -f $D/pwr.pid" >/dev/null 2>&1; }
trap 'stop_sampler; kill_benches || true; wake_release || true' EXIT
wake_hold 3600 || exit 1
screen_off || exit 1

adb push -q /tmp/npu-spike/phone/npu_bench $D/ >/dev/null
adb push -q /tmp/npu-spike/out/kbias_w800.f32 $D/ >/dev/null
for v in f32mask fp16wmask wi8mask; do
  adb push -q /tmp/npu-spike/out/qwen3_enc_w800_$v.tflite $D/ >/dev/null
done
adb shell "mkdir -p $D/sbenc2 $D/cache"
adb push -q /tmp/npu-spike/phone/sbenc2/. $D/sbenc2/ >/dev/null

# ---- latency + outputs of the masked variants ----
for v in f32mask fp16wmask wi8mask; do
  echo "===== $v npu"
  timeout 900 adb shell "cd $D && LD_LIBRARY_PATH=$D/lib NPU_CACHE_DIR=$D/cache ./npu_bench $D/qwen3_enc_w800_$v.tflite npu $D/lib 20 $D/mel_w800.f32 $D/kbias_w800.f32 -- $D/out/${v}_npu" > "$LOG/${v}_npu.txt" 2>&1
  grep -E "compile|first run|runs |FAIL" "$LOG/${v}_npu.txt"
  adb pull -q $D/out/${v}_npu_out0.f32 "$LOG/" >/dev/null 2>&1
done
echo "===== f32mask npu (cached compile)"
timeout 900 adb shell "cd $D && LD_LIBRARY_PATH=$D/lib NPU_CACHE_DIR=$D/cache ./npu_bench $D/qwen3_enc_w800_f32mask.tflite npu $D/lib 20 $D/mel_w800.f32 $D/kbias_w800.f32" > "$LOG/f32mask_npu_cached.txt" 2>&1
grep -E "load|compile|first run|runs |FAIL" "$LOG/f32mask_npu_cached.txt"

# ---- energy ----
adb shell "dumpsys battery | grep -E 'status|level|Charge counter|AC powered|USB powered|Wireless powered'" | tr -d '\r' | tee "$LOG/battery.txt"
grep -qE 'status: 3$' "$LOG/battery.txt" || { echo "NOT ON BATTERY (unplug the phone): skipping energy"; echo done; exit 0; }
adb shell "nohup sh -c 'while :; do echo \$(date +%s.%N) \$(cat /sys/class/power_supply/battery/current_now) \$(cat /sys/class/power_supply/battery/voltage_now); sleep 0.05; done' > $D/pwr.txt 2>/dev/null & echo \$! > $D/pwr.pid"
stamp() { echo "$1 $(adb shell 'date +%s.%N' | tr -d '\r')" >> "$LOG/stamps.txt"; }
: > "$LOG/stamps.txt"
sleep 5
stamp idle0_start; sleep 30; stamp idle0_end
for rep in 1 2; do
  timeout 900 adb shell "cd $D && LD_LIBRARY_PATH=$D/lib NPU_CACHE_DIR=$D/cache ./npu_bench $D/qwen3_enc_w800_f32mask.tflite npu $D/lib 2000 $D/mel_w800.f32 $D/kbias_w800.f32" > "$LOG/energy_npu_$rep.txt" 2>&1
  grep -E "runs |loop_epoch|FAIL" "$LOG/energy_npu_$rep.txt"
  sleep 10
  wait_benches || true
  for n in 10 110; do
    stamp gpu${rep}_n${n}_start
    timeout 900 adb shell "cd $D/sbenc2 && LD_LIBRARY_PATH=$D/sbenc2 NPU_SPIKE_ENC_ONLY=1 STARLING_ENGINE=fast STARLING_FAST_TIMING=1 STARLING_FAST_CACHE_DIR=$D/sbenc2 ./starling-bench --model qwen3 --gguf $G --warmup --runs $n --quiet short.wav" > "$LOG/energy_gpu_${rep}_$n.txt" 2>&1
    stamp gpu${rep}_n${n}_end
    grep -c "enc+prefill" "$LOG/energy_gpu_${rep}_$n.txt"
    grep "enc+prefill" "$LOG/energy_gpu_${rep}_$n.txt" | tail -2 | sed 's/decode=.*total/ total/'
    wait_benches || true
    sleep 10
  done
done
stamp idle1_start; sleep 30; stamp idle1_end
stop_sampler
adb pull -q $D/pwr.txt "$LOG/pwr.txt" >/dev/null
wc -l "$LOG/pwr.txt"
adb shell "dumpsys thermalservice | grep -m2 -iE 'Temperature\{'" 2>/dev/null
echo done
