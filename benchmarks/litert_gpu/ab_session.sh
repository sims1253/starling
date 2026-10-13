#!/bin/bash
# Old vs new encoder/prefill GEMM on the Pixel 10 Pro, one binary:
# STARLING_FAST_SG=0 (gemm.comp) vs STARLING_FAST_SG=1 (gemm_sg.comp).
#
#   flock /tmp/starling-phone.lock benchmarks/litert_gpu/ab_session.sh LOGDIR PHASE...
#
#   push            BIN (dir with starling-bench + libggml*.so) -> $D
#   micro           gemmx probes (STARLING_FAST_MICRO) at the real shapes, g vs f
#   parity          every model x short/medium, SG=0 and SG=1, transcripts
#   ab              every model: interleaved rounds (0 1 1 0) x short/medium,
#                   RUNS timed runs each (enc+prefill, decode, wall)
#   nb              gemmx at the small-N shapes with 8 vs 16 columns per workgroup
#   ablate          Qwen3-0.6B short, both arms: STARLING_FAST_SKIP per GEMM class
#   energy          (on battery only) charge counter over N transcriptions per arm
# MODELS overrides the model list ("slug:gguf ..."). Models and fixtures are
# read from /data/local/tmp/starling; only $D is written (+ the wake-hold dex).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.178.59:45929}
LOG=${1:?LOGDIR}; shift
mkdir -p "$LOG"
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../fast_engine/phone_common.sh"
export ANDROID_HOME=${ANDROID_HOME:-$HOME/android-sdk}
D=${DEV:-/data/local/tmp/litert-gpu/sg}
S=/data/local/tmp/starling
BIN=${BIN:-/tmp/litert-gpu/bin/sg1}
RUNS=${RUNS:-3}
MODELS=${MODELS:-"qwen3:qwen3-asr-0.6b-q4e8.gguf qwen3:qwen3-asr-1.7b-q4e8-imx.gguf moss:moss-transcribe-preview-2b-q4e8-fullimx.gguf ark06:ark-asr-0.6b-q4k.gguf granite:granite-2b-dynq4.imx.gguf voxtral:voxtral-mini-4b-realtime-q4k.gguf parakeet:parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf"}

state=$(wakefulness); echo "display: $state"
if [ "$state" = "mWakefulness=Awake" ]; then
  fg=$(adb shell "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'" | tr -d '\r')
  case "$fg" in *Launcher*|*launcher*|*SystemUI*|*keyguard*|"") ;; *) echo "PHONE IN USE: $fg"; exit 3 ;; esac
fi
trap 'kill_benches || true; wake_release || true' EXIT
wake_hold 7200 || exit 1
screen_off || exit 1

# run_sb TAG SG WAV RUNS [EXTRA_ENV] [QUIET]: one starling-bench invocation.
run_sb() {
  local tag=$1 sg=$2 wav=$3 runs=$4 extra=${5:-} q=${6---quiet}
  wait_benches || true
  timeout 2400 adb shell "cd $D && LD_LIBRARY_PATH=$D STARLING_ENGINE=fast STARLING_GGML_THREADS=6 STARLING_FAST_TIMING=1 STARLING_FAST_CACHE_DIR=$D/cache STARLING_FAST_SG=$sg $extra ./starling-bench --model $slug --gguf $S/$gguf --warmup --runs $runs $q $S/$wav" > "$LOG/$tag.txt" 2>&1
  echo "exit $?" >> "$LOG/$tag.txt"
}

for phase in "$@"; do
  echo "######## $phase $(date +%T)"
  case $phase in
  push)
    adb shell "mkdir -p $D/cache"
    for f in "$BIN"/*; do adb push -q "$f" "$D/" >/dev/null || echo "push $f failed"; done
    adb shell "chmod 755 $D/starling-bench; sha256sum $D/starling-bench"
    ;;
  micro)
    slug=qwen3; gguf=qwen3-asr-0.6b-q4e8.gguf
    M=""
    for v in g f; do
      for s in 104,896,896 104,3584,896 104,896,3584 104,1024,7680 112,4096,1024 112,1024,2048 112,6144,1024 112,1024,3072 320,6144,1024 320,1024,3072; do
        for k in w4 w8 f16; do M="$M;gemmx,$s,$k,16,$v"; done
      done
    done
    run_sb micro 1 short.wav 1 "STARLING_FAST_MICRO='${M#;}'"
    grep -E "fast-micro|rror" "$LOG/micro.txt"
    ;;
  parity)
    for e in $MODELS; do
      slug=${e%%:*}; gguf=${e#*:}
      for wav in short.wav medium.wav; do
        for sg in 0 1; do
          run_sb "par_${gguf%.gguf}_${wav%.wav}_sg$sg" $sg $wav 1 "" ""
        done
        a=$(grep -E '^  ' "$LOG/par_${gguf%.gguf}_${wav%.wav}_sg0.txt")
        b=$(grep -E '^  ' "$LOG/par_${gguf%.gguf}_${wav%.wav}_sg1.txt")
        [ "$a" = "$b" ] && echo "SAME  ${gguf%.gguf} $wav" || echo "DIFF  ${gguf%.gguf} $wav"
      done
    done
    ;;
  ab)
    for e in $MODELS; do
      slug=${e%%:*}; gguf=${e#*:}
      for wav in short.wav medium.wav; do
        r=0
        for sg in 0 1 1 0; do
          r=$((r + 1))
          run_sb "ab_${gguf%.gguf}_${wav%.wav}_r${r}_sg$sg" $sg $wav "$RUNS"
          echo "${gguf%.gguf} $wav r$r sg$sg: $(grep -oE 'enc\+prefill=[0-9.]+ms|decode=[0-9.]+ms \([0-9]+ tokens|total=[0-9.]+ms|encoder=[0-9.]+ms' "$LOG/ab_${gguf%.gguf}_${wav%.wav}_r${r}_sg$sg.txt" | tail -3 | tr '\n' ' ') $(grep -E 'exit [1-9]' "$LOG/ab_${gguf%.gguf}_${wav%.wav}_r${r}_sg$sg.txt")"
        done
      done
    done
    ;;
  nb)
    slug=qwen3; gguf=qwen3-asr-0.6b-q4e8.gguf
    for nbv in 8 16; do
      M=""
      for s in 104,896,896 104,1024,1024 104,1024,7680 112,1024,2048 112,2048,1024 112,1024,1024 320,1024,1024; do
        for k in w4 w8; do M="$M;gemmx,$s,$k,16,g"; done
      done
      run_sb "nb$nbv" 1 short.wav 1 "STARLING_FAST_SG_NB=$nbv STARLING_FAST_MICRO='${M#;}'"
      grep -E "fast-micro|rror" "$LOG/nb$nbv.txt" | sed "s/^/NB$nbv /"
    done
    ;;
  ablate)
    slug=qwen3; gguf=qwen3-asr-0.6b-q4e8.gguf
    for sg in 0 1 1 0; do
      for cls in none conv enclin llmlin attn; do
        case $cls in
          none) sk="" ;;
          conv) sk="enc_conv2,enc_conv3" ;;
          enclin) sk="enc_qkv,enc_attn_o,enc_ff_up,enc_ff_down,enc_proj" ;;
          llmlin) sk="llm_qkv,llm_attn_o,llm_ff_up,llm_ff_down" ;;
          attn) sk="enc_attn_qk,enc_attn_pv,llm_attn_qk,llm_attn_pv" ;;
        esac
        run_sb "abl_sg${sg}_$cls" $sg short.wav 3 "STARLING_FAST_SKIP='$sk'"
        echo "sg$sg skip=$cls: $(grep -oE 'enc\+prefill=[0-9.]+ms' "$LOG/abl_sg${sg}_$cls.txt" | tail -3 | tr '\n' ' ')"
        cp "$LOG/abl_sg${sg}_$cls.txt" "$LOG/abl_sg${sg}_${cls}_$(date +%H%M%S).txt"
      done
    done
    ;;
  energy)
    adb shell "dumpsys battery" | tr -d '\r' > "$LOG/battery.txt"
    grep -qE 'status: 3$' "$LOG/battery.txt" || { echo "on charger: skipping energy"; continue; }
    for e in $MODELS; do
      slug=${e%%:*}; gguf=${e#*:}
      for sg in 0 1 1 0; do
        c0=$(adb shell cat /sys/class/power_supply/battery/charge_counter | tr -d '\r'); t0=$(date +%s.%N)
        run_sb "en_${gguf%.gguf}_sg${sg}_$c0" $sg short.wav ${ENERGY_RUNS:-10}
        c1=$(adb shell cat /sys/class/power_supply/battery/charge_counter | tr -d '\r'); t1=$(date +%s.%N)
        echo "${gguf%.gguf} sg$sg: charge $c0 -> $c1 uAh over $(echo "$t1 - $t0" | bc) s" | tee -a "$LOG/energy.txt"
      done
    done
    ;;
  *) echo "unknown phase $phase" ;;
  esac
done
echo "done $(date +%T)"
