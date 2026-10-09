#!/usr/bin/env bash
# wedge_forensics.sh — capture the evidence of a GPU wedge or a spontaneous
# phone restart.
#
#   wedge_forensics.sh watch start|stop                    # around a phone session
#   wedge_forensics.sh event <label>                       # the phone is still up
#   wedge_forensics.sh post-reboot <label> [--bugreport]   # it restarted
#
# `watch start` keeps a rotating device-side logcat (all buffers) for the
# session: pixel-thermal floods the main buffer within a minute, so a later
# capture has lost the minutes that matter. `event`/`post-reboot` pull it.
# Capture before any retry, recovery load or reboot. Output goes under
# $STARLING_FORENSICS_DIR (default ~/starling-forensics). Every adb call is
# bounded; ANDROID_SERIAL picks the device. Procedure:
# benchmarks/fast_engine/AUTORESEARCH.md, "GPU failures and phone restarts".
set -uo pipefail

mode=${1:-}
label=${2:-event}
bugreport=0
[ "${3:-}" = "--bugreport" ] && bugreport=1
DEV=/data/local/tmp/starling
APP=dev.starling.mobile
WATCH=$DEV/forensics-logcat   # rotating files: logcat.txt, logcat.txt.1, ...

case "$mode" in
  watch)
    case "$label" in start|stop) ;; *) echo "usage: $0 watch start|stop" >&2; exit 2 ;; esac
    # Stop any watcher first, so `start` never leaves two running. ([l]: the
    # pattern must not match the shell running pkill.)
    timeout 15 adb shell "pkill -f '[l]ogcat -b all -v threadtime -f $WATCH' ; true" >/dev/null 2>&1
    if [ "$label" = stop ]; then
      echo "logcat watch stopped (files stay in $WATCH until the next start)"
    elif timeout 15 adb shell "mkdir -p $WATCH && (logcat -b all -v threadtime -f $WATCH/logcat.txt -r 16384 -n 8 \
        </dev/null >/dev/null 2>$WATCH/stderr.txt & echo \$! > $WATCH/pid) && sleep 1 && kill -0 \$(cat $WATCH/pid)"; then
      echo "logcat watch running (device: $WATCH, 8 x 16 MB)"
    else
      echo "ERROR: the logcat watch did not start:" >&2
      timeout 15 adb shell "cat $WATCH/stderr.txt" >&2
      exit 1
    fi
    exit 0 ;;
  event|post-reboot) ;;
  *) echo "usage: $0 watch start|stop | event|post-reboot <label> [--bugreport]" >&2; exit 2 ;;
esac
adb get-state >/dev/null 2>&1 || { echo "no adb device reachable" >&2; exit 1; }
out="${STARLING_FORENSICS_DIR:-$HOME/starling-forensics}/$(date +%Y%m%d-%H%M%S)-$mode-$label"
mkdir -p "$out"

sh_() {   # sh_ <seconds> <command>: bounded adb shell, CR stripped
  timeout "$1" adb shell "$2" 2>&1 | tr -d '\r'
}

{
  echo "== $mode: $label  (host $(date -Is))"
  echo "-- device date / uptime / load:"
  sh_ 20 "date; uptime"
  echo "-- boot reasons (a spontaneous restart is anything but reboot,shell / reboot,userrequested):"
  sh_ 20 'for p in sys.boot.reason sys.boot.reason.last ro.boot.bootreason persist.sys.boot.reason.history; do echo "$p=$(getprop $p | tr "\n" " ")"; done'
  echo "-- build:"
  sh_ 20 "getprop ro.build.fingerprint; uname -r"
  echo "-- wedge marker (bench dir; the app's lives in its private storage):"
  sh_ 20 "cat $DEV/starling-fast-gpu-wedged 2>/dev/null && stat -c 'mtime: %y' $DEV/starling-fast-gpu-wedged 2>/dev/null || echo none"
  echo "-- processes (benches, app):"
  sh_ 20 "pidof starling-bench starling-bench-base starling-bench-cand || echo no-bench; pidof $APP || echo app-not-running"
  echo "-- power / doze state (a suspended or dozing phone stalls GPU round-trips):"
  sh_ 20 "dumpsys power | grep -E 'mWakefulness=|mStayOn=|mIsPowered=|mHoldingWakeLockSuspendBlocker=|mHoldingDisplaySuspendBlocker=' | sort -u; dumpsys deviceidle | grep -E 'mState=|mLightState=' | head -2"
  echo "-- battery:"
  sh_ 20 "dumpsys battery | grep -E 'AC powered|USB powered|Wireless powered|status|level|temperature|Charge counter'"
  echo "-- thermal:"
  sh_ 20 "dumpsys thermalservice | sed -n '1,20p'"
  echo "-- GPU devfreq:"
  sh_ 20 'for d in /sys/class/devfreq/*gpu*; do echo "$d"; for f in cur_freq min_freq max_freq; do echo "  $f=$(cat $d/$f 2>/dev/null)"; done; tail -4 $d/trans_stat 2>/dev/null; done'
  echo "-- memory:"
  sh_ 20 "grep -E 'MemTotal|MemFree|MemAvailable|ION|DmaBuf|GPU|KReclaimable' /proc/meminfo"
} > "$out/summary.txt"

# Kernel log: the GPU driver (PowerVR: pvr/rogue, hardware recovery = HWR,
# lockup, firmware) and watchdog lines. On a post-reboot capture this is the
# NEW boot's log — the previous boot's survives only in the bugreport / the
# dropbox last-kmsg entries below.
sh_ 60 "logcat -b kernel -d" > "$out/kernel.txt"
sh_ 60 "logcat -d -b main,system,crash" > "$out/logcat.txt"
if timeout 15 adb shell "ls $WATCH/logcat.txt" >/dev/null 2>&1; then
  timeout 300 adb pull "$WATCH" "$out/watch" >/dev/null 2>&1 || echo "watch pull failed" >&2
fi
grep -rhiE "pvr|rogue|rgx|hwr|lockup|gpu|watchdog|panic|vulkan|starling|lowmemorykiller|lmkd" \
  "$out/kernel.txt" "$out/logcat.txt" "$out/watch" 2>/dev/null | sort -u > "$out/gpu-lines.txt" || true
# GPU power rail, logged once a minute by pixel-thermal: near 0 mW during a
# blocked fence wait means the GPU sat idle with work outstanding (lost work,
# not a long job).
grep -rh "S2S_VDD_GPU" "$out/logcat.txt" "$out/watch" 2>/dev/null | sort -u |
  sed -nE 's/^([0-9-]+ [0-9:.]+).*(S2S_VDD_GPU: [0-9.]+ mW).*(VSYS_PWR_VBATT: [0-9.]+ mW).*/\1  \2  \3/p' \
  > "$out/gpu-rail.txt" || true

if [ "$mode" = post-reboot ]; then
  {
    echo "-- dropbox entries (newest last):"
    sh_ 30 "dumpsys dropbox" | grep -E "SYSTEM_|kernel|panic|watchdog|tombstone|gpu" | tail -30
    for tag in SYSTEM_LAST_KMSG SYSTEM_RESTART system_server_watchdog SYSTEM_TOMBSTONE; do
      echo "== $tag"
      sh_ 60 "dumpsys dropbox --print $tag" | tail -200
    done
  } > "$out/dropbox.txt"
  if [ "$bugreport" = 1 ]; then
    echo "capturing bugreport (several minutes)..."
    timeout 900 adb bugreport "$out/" > "$out/bugreport.log" 2>&1 || echo "bugreport failed: see bugreport.log"
  fi
fi

echo "forensics: $out"
grep -A3 "boot reasons" "$out/summary.txt" | tail -3
echo "GPU rail samples: $(wc -l < "$out/gpu-rail.txt") (gpu-rail.txt); watch log: $([ -d "$out/watch" ] && echo pulled || echo none — run 'watch start' before sessions)"
echo "GPU/driver lines: $(wc -l < "$out/gpu-lines.txt") (gpu-lines.txt); marker: $(sed -n '/wedge marker/{n;p}' "$out/summary.txt")"
