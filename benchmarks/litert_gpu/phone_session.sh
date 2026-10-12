#!/bin/bash
# LiteRT GPU (ML Drift) vs Starling's fast Vulkan engine on the Pixel 10 Pro.
#
#   flock /tmp/starling-phone.lock benchmarks/litert_gpu/phone_session.sh LOGDIR PHASE...
#
# Phases:
#   push    stage bench, LiteRT libs, clshim, encoder .tflite variants, micro models,
#           starling-bench (+ ggml libs, short.wav, a private copy of the GGUF)
#   probe   one verbose GPU run (which backend / precision LiteRT picks), logcat
#   enc     encoder window: {f32mask, fp16wmask, wi8mask} x {fp16, fp32}, RUNS runs
#   kdump   encoder under clshim: kernel sources + per-kernel GPU time (fp16, fp32);
#           NOCB=1 (default) hides cl_khr_command_buffer so every kernel gets an event
#   micro   FC micro models + one Qwen3 layer under clshim (fp16, fp32)
#   alu     OpenCL ALU probes (cl_alu_probe.c): f32 vs f16 FMA rate, broadcast loop
#   ours    starling-bench: full, encoder-only by ablation, prefill-only, gemmx probes
#
# Only /data/local/tmp/litert-gpu is written (plus the shared wake-hold dex).
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.178.59:45929}
LOG=${1:?LOGDIR}; shift
mkdir -p "$LOG"
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../fast_engine/phone_common.sh"
BENCH_BINS="$BENCH_BINS litert_gpu_bench cl_alu_probe"
export ANDROID_HOME=${ANDROID_HOME:-$HOME/android-sdk}

D=/data/local/tmp/litert-gpu
STAGE=${STAGE:-/tmp/litert-gpu/phone}
ENC=${ENC:-/tmp/npu-spike/out}
MICRO=${MICRO:-/tmp/litert-gpu/micro}
SB=${SB:-/home/m0hawk/.t3/worktrees/starling/t3-1d904395/build-android}
WAV=${WAV:-$HERE/../../tests/fixtures/2086-149220-0033.wav}
GSRC=/data/local/tmp/starling/qwen3-asr-0.6b-q4e8.gguf
G=$D/sb/qwen3-asr-0.6b-q4e8.gguf
RUNS=${RUNS:-20}

state=$(wakefulness); echo "display: $state"
if [ "$state" = "mWakefulness=Awake" ]; then
  fg=$(adb shell "dumpsys activity activities | grep -m1 -E 'mResumedActivity|topResumedActivity'" | tr -d '\r')
  case "$fg" in *Launcher*|*launcher*|*SystemUI*|*keyguard*|"") ;; *) echo "PHONE IN USE: $fg"; exit 3 ;; esac
fi
trap 'kill_benches || true; wake_release || true' EXIT
wake_hold 7200 || exit 1
screen_off || exit 1

dsize() { adb shell "stat -c %s $1 2>/dev/null" | tr -d '\r'; }
push_if() {  # push_if LOCAL REMOTE: skip when the sizes match
  local s; s=$(stat -c %s "$1")
  [ "$(dsize "$2")" = "$s" ] || timeout 1200 adb push -q "$1" "$2" >/dev/null || { echo "push $1 failed"; return 1; }
}
dev_time() { adb shell 'date +%m-%d\ %H:%M:%S.000' | tr -d '\r'; }
temps() { adb shell "dumpsys thermalservice 2>/dev/null | grep -m3 -E 'mValue=.*(GPU|gpu|VIRTUAL-SKIN|soc)'" | tr -d '\r' | sed 's/^ *//' | head -3; }

# run_lt TAG MODEL TOML RUNS [SHIM] [INPUTS...]
run_lt() {
  local tag=$1 model=$2 toml=$3 runs=$4 shim=${5:-} ; shift 5
  local ins="$*" lp="$D/bin/lib" env=""
  if [ -n "$shim" ]; then
    adb shell "rm -rf $D/kdump/$tag; mkdir -p $D/kdump/$tag"
    lp="$D/bin/shim:$D/bin/lib"
    env="CLSHIM_DIR=$D/kdump/$tag CLSHIM_PROFILE=${PROFILE:-1} CLSHIM_NOCB=${NOCB:-1}"
  fi
  [ -n "${LITERT_LOG:-}" ] && env="$env LITERT_LOG=$LITERT_LOG"
  wait_benches || true
  echo "===== $tag ($toml)"
  timeout 1200 adb shell "cd $D && LD_LIBRARY_PATH=$lp $env LITERT_GPU_TOML='$toml' ./bin/litert_gpu_bench $model gpu $D/bin/lib $runs $ins -- $D/out/$tag" > "$LOG/$tag.txt" 2>&1
  echo "exit $?" >> "$LOG/$tag.txt"
  grep -E "compile|first run|runs |FAIL|exit [1-9]" "$LOG/$tag.txt"
  if [ -n "$shim" ]; then
    mkdir -p "$LOG/kdump/$tag"
    adb pull -q $D/kdump/$tag/. "$LOG/kdump/$tag/" >/dev/null 2>&1
    grep -E "^== |clshim:" "$LOG/kdump/$tag/clshim.log" | head -4
  fi
  adb pull -q $D/out/${tag}_out0.f32 "$LOG/" >/dev/null 2>&1 || true
}

for phase in "$@"; do
  echo "######## $phase  $(date +%T)  $(temps | tr '\n' ' ')"
  case $phase in
  push)
    adb shell "mkdir -p $D/bin/lib $D/bin/shim $D/models $D/micro $D/in $D/out $D/kdump $D/sb/cache"
    push_if "$STAGE/litert_gpu_bench" $D/bin/litert_gpu_bench
    adb shell "chmod 755 $D/bin/litert_gpu_bench"
    for f in "$STAGE"/lib/*.so; do push_if "$f" $D/bin/lib/$(basename "$f"); done
    for f in "$STAGE"/shim/*.so; do push_if "$f" $D/bin/shim/$(basename "$f"); done
    for v in f32mask fp16wmask wi8mask; do push_if "$ENC/qwen3_enc_w800_$v.tflite" $D/models/enc_$v.tflite; done
    push_if "$ENC/mel_w800.f32" $D/in/mel_w800.f32
    push_if "$ENC/kbias_w800.f32" $D/in/kbias_w800.f32
    for f in "$MICRO"/*.tflite; do push_if "$f" $D/micro/$(basename "$f"); done
    push_if "$SB/starling-bench" $D/sb/starling-bench
    for f in "$SB"/ggml/src/libggml*.so; do push_if "$f" $D/sb/$(basename "$f"); done
    push_if "$WAV" $D/sb/short.wav
    adb shell "chmod 755 $D/sb/starling-bench; [ -f $G ] || cp $GSRC $G; ls -la $D/sb $D/bin $D/models | head -40"
    ;;
  probe)
    t0=$(dev_time)
    LITERT_LOG=0 run_lt probe_fp16w $D/models/enc_fp16wmask.tflite "" 3 "" $D/in/mel_w800.f32 $D/in/kbias_w800.f32
    adb logcat -d -T "$t0" 2>/dev/null | grep -iE "litert|drift|opencl|gpu|tflite|delegate" > "$LOG/probe_logcat.txt"
    head -40 "$LOG/probe_fp16w.txt"
    ;;
  enc)
    for v in f32mask fp16wmask wi8mask; do
      for p in 1 2; do
        run_lt enc_${v}_p$p $D/models/enc_$v.tflite "precision = $p" "$RUNS" "" $D/in/mel_w800.f32 $D/in/kbias_w800.f32
      done
    done
    ;;
  backends)
    for b in 3 2; do
      run_lt enc_f32mask_p1_b$b $D/models/enc_f32mask.tflite "precision = 1; backend = $b" 10 "" $D/in/mel_w800.f32 $D/in/kbias_w800.f32
    done
    ;;
  kdump)
    for p in 1 2; do
      run_lt kd_enc_p$p $D/models/enc_f32mask.tflite "precision = $p" 5 shim $D/in/mel_w800.f32 $D/in/kbias_w800.f32
    done
    ;;
  micro)
    for f in "$MICRO"/*.tflite; do
      b=$(basename "$f" .tflite)
      for p in 1 2; do run_lt mi_${b}_p$p $D/micro/$b.tflite "precision = $p" 20 shim -; done
    done
    ;;
  ours)
    cd_sb="cd $D/sb && LD_LIBRARY_PATH=$D/sb STARLING_ENGINE=fast STARLING_FAST_TIMING=1 STARLING_FAST_CACHE_DIR=$D/sb/cache"
    sb_run() {  # sb_run TAG ENV RUNS
      wait_benches || true
      echo "===== ours $1 ($2)"
      timeout 1500 adb shell "$cd_sb $2 ./starling-bench --model qwen3 --gguf $G --warmup --runs $3 --quiet short.wav" > "$LOG/ours_$1.txt" 2>&1
      echo "exit $?" >> "$LOG/ours_$1.txt"
      grep -E "enc\+prefill|fast-micro|exit [1-9]|rror" "$LOG/ours_$1.txt" | tail -${4:-8}
    }
    sb_run full "" 6
    sb_run enconly "STARLING_FAST_SKIP='llm_*,rope_kv,lm_head'" 6
    sb_run prefillonly "STARLING_FAST_SKIP='enc_*'" 6
    sb_run full2 "" 6
    M="gemmx,104,4096,1024,f16,16;gemmx,104,896,896,f16,16;gemmx,104,3584,896,f16,16;gemmx,104,896,3584,f16,16"
    M="$M;gemmx,112,4096,1024,f16,16;gemmx,112,1024,2048,f16,16;gemmx,112,6144,1024,f16,16;gemmx,112,1024,3072,f16,16"
    M="$M;gemmx,320,4096,1024,f16,16;gemmx,320,6144,1024,f16,16;gemmx,320,1024,3072,f16,16"
    M="$M;gemmx,112,4096,1024,w4,16;gemmx,112,6144,1024,w4,16;gemmx,104,3584,896,w8,16;gemmx,104,896,3584,w8,16"
    sb_run micro "STARLING_FAST_MICRO='$M'" 1 20
    ;;
  alu)
    push_if "$STAGE/cl_alu_probe" $D/bin/cl_alu_probe
    adb shell "chmod 755 $D/bin/cl_alu_probe"
    for g in 512 2048; do
      wait_benches || true
      timeout 300 adb shell "$D/bin/cl_alu_probe $g 4096" 2>&1 | tee "$LOG/alu_$g.txt"
    done
    ;;
  *) echo "unknown phase $phase" ;;
  esac
done
temps
echo "done $(date +%T)"
