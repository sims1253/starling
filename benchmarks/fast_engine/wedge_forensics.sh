#!/usr/bin/env bash
# wedge_forensics.sh — capture the evidence for a GPU wedge or a spontaneous
# phone restart (#325) so the root cause can eventually be identified.
#
#   wedge_forensics.sh event <label>                  # the phone is still up
#   wedge_forensics.sh post-reboot <label> [--bugreport]   # it restarted
#
# Run it FIRST, before any retry, recovery load or reboot: the kernel log is
# a ring buffer (GPU driver lines rotate out within hours) and a reboot
# replaces it. Writes a directory under $STARLING_FORENSICS_DIR (default
# ~/starling-forensics) and prints a one-screen summary. Every adb call is
# bounded. Uses the adb device from ANDROID_SERIAL when several are attached.
# What to do with the result: benchmarks/fast_engine/AUTORESEARCH.md,
# "GPU failures and phone restarts".
set -uo pipefail

mode=${1:-}
label=${2:-event}
bugreport=0
[ "${3:-}" = "--bugreport" ] && bugreport=1
case "$mode" in
  event|post-reboot) ;;
  *) echo "usage: $0 event|post-reboot <label> [--bugreport]" >&2; exit 2 ;;
esac

DEV=/data/local/tmp/starling
APP=dev.starling.mobile
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
grep -iE "pvr|rogue|rgx|hwr|lockup|gpu|watchdog|panic|vulkan|starling|lowmemorykiller|lmkd" \
  "$out/kernel.txt" "$out/logcat.txt" > "$out/gpu-lines.txt" || true

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
echo "GPU/driver lines: $(wc -l < "$out/gpu-lines.txt") (gpu-lines.txt); marker: $(sed -n '/wedge marker/{n;p}' "$out/summary.txt")"
