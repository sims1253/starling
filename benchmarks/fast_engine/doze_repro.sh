#!/usr/bin/env bash
# doze_repro.sh — does GPU work on a dozing phone wedge the driver? (#325)
#
#   doze_repro.sh [arms]      # default arms: "D A D A D A"
#
# Alternates trials of the same workload with the phone in deep Doze (D:
# screen off + `dumpsys deviceidle force-idle`, no wake signals) and held
# awake (A: Doze lifted, screen woken every poll), or dozing under the
# benches' shell wake lock (W: as D, plus phone_common.sh `wake_hold`; does
# the lock alone prevent the wedge?). Each trial is one bench
# process: load / run / free the model 3 times (`--cycles 3 --runs 2`).
# After a wedge it captures forensics (wedge_forensics.sh event) and waits
# out the 15-minute wedge marker before the next trial. Stops after 3
# wedges, or on a restart (the kernel boot id changed: post-reboot forensics,
# then stop) — a dozing phone may just drop wifi adb for a while, which only
# means waiting. Always restores normal Doze and wakes the screen.
#
# Needs: a Pixel on adb (ANDROID_SERIAL when several devices), UNPLUGGED
# (Doze never engages on power) — or FAKE_UNPLUG=1: the framework is told the
# phone is unplugged (`dumpsys battery unplug`, reset on exit) while it stays
# on the charger; run D arms alongside as the control that the wedge still
# reproduces that way, the bench binary + libs in $DEV/lt and the
# model/fixture in $DEV. Every adb call is bounded. Results: one line per
# trial on stdout and in $OUT/trials.txt.
set -uo pipefail

arms=${1:-"D A D A D A"}
DEV=/data/local/tmp/starling
GGUF=${GGUF:-parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf}
WAV=${WAV:-medium.wav}
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/phone_common.sh"
OUT=${STARLING_FORENSICS_DIR:-$HOME/starling-forensics}/doze-repro-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
export STARLING_FORENSICS_DIR=$OUT

sh_() { timeout "$1" adb shell "$2" 2>/dev/null | tr -d '\r'; }
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/trials.txt"; }
state() { sh_ 15 "dumpsys power | grep -m1 -o 'mWakefulness=[A-Za-z]*'; dumpsys deviceidle | grep -m1 -o 'mState=[A-Z_]*'; dumpsys battery | grep -E '^  (level|USB powered|AC powered):' | tr -d ' ' | tr '\n' ' '" | tr '\n' ' '; }
restore() {
  wake_release || true
  [ -n "${FAKE_UNPLUG:-}" ] && sh_ 15 "dumpsys battery reset" >/dev/null
  sh_ 15 "dumpsys deviceidle unforce; input keyevent KEYCODE_WAKEUP" >/dev/null
}
trap restore EXIT

# Wifi adb drops while the phone dozes and comes back on a new port: on a
# failed probe, rediscover the phone over mDNS and reconnect (wireless
# debugging serials only; a USB serial is left alone).
reachable() {
  timeout 15 adb shell true >/dev/null 2>&1 && return 0
  case "${ANDROID_SERIAL:-}" in *:*|*_adb-tls-connect*) ;; *) return 1 ;; esac
  local ep
  ep=$(timeout 20 adb mdns services 2>/dev/null | awk '/_adb-tls-connect/ {print $NF; exit}')
  [ -n "$ep" ] || return 1
  timeout 25 adb connect "$ep" >/dev/null 2>&1 || return 1
  export ANDROID_SERIAL=$ep
  timeout 15 adb shell true >/dev/null 2>&1
}
boot_id() { sh_ 15 "cat /proc/sys/kernel/random/boot_id"; }
marker_fresh() {   # the engine refuses for 15 min after a wedge
  local m; m=$(sh_ 15 "stat -c %Y $DEV/starling-fast-gpu-wedged 2>/dev/null")
  [ -n "$m" ] && [ $(( $(sh_ 15 'date +%s') - m )) -lt 930 ]
}

reachable || { echo "phone not reachable" >&2; exit 1; }
[ -n "${FAKE_UNPLUG:-}" ] && { sh_ 15 "dumpsys battery unplug" >/dev/null; log "FAKE_UNPLUG: framework told the phone is unplugged"; }
sh_ 15 "dumpsys battery" | grep -qE "(AC|USB|Wireless) powered: true" &&
  { echo "phone is on power: Doze never engages — unplug it first" >&2; exit 1; }
"$HERE/wedge_forensics.sh" watch start >/dev/null
boot=$(boot_id)
[ -n "$boot" ] || { echo "phone not reachable" >&2; exit 1; }
wedges=0 n=0
for arm in $arms; do
  n=$((n + 1))
  until reachable; do sleep 20; done
  while marker_fresh; do sleep 30; reachable >/dev/null; done
  sh_ 15 "pidof starling-bench >/dev/null && echo busy" | grep -q busy && { log "trial $n: a bench is still running — stop"; break; }
  wake_release || true
  if [ "$arm" = W ]; then
    wake_hold 1800 || { log "trial $n [W]: could not take the wake lock — stop"; break; }
  fi
  if [ "$arm" = D ] || [ "$arm" = W ]; then
    sh_ 15 "input keyevent KEYCODE_SLEEP" >/dev/null; sleep 5
    sh_ 15 "dumpsys deviceidle force-idle" >/dev/null; sleep 5
  else
    sh_ 15 "dumpsys deviceidle unforce; input keyevent KEYCODE_WAKEUP" >/dev/null; sleep 3
  fi
  before=$(state)
  sh_ 15 "rm -f $DEV/lt/trial.txt"
  sh_ 20 "cd $DEV && (timeout 900 env LD_LIBRARY_PATH=$DEV/lt STARLING_ENGINE=fast STARLING_GGML_THREADS=6 \
    STARLING_FAST_CACHE_DIR=$DEV ./lt/starling-bench --model parakeet --gguf $DEV/$GGUF --cycles 3 --runs 2 \
    --quiet $DEV/$WAV > $DEV/lt/trial.txt 2>&1; echo exit=\$? >> $DEV/lt/trial.txt) </dev/null >/dev/null 2>&1 &"
  t0=$(date +%s) res="" restarted=0
  while :; do   # the bench bounds itself (timeout 900 on the device)
    [ "$arm" = A ] && sh_ 10 "input keyevent KEYCODE_WAKEUP" >/dev/null
    if reachable; then
      [ "$(boot_id)" != "$boot" ] && { restarted=1; break; }
      res=$(sh_ 15 "cat $DEV/lt/trial.txt")
      echo "$res" | grep -q "^exit=" && break
    fi
    [ $(( $(date +%s) - t0 )) -gt 3600 ] && { res="host gave up after 1 h"; break; }
    sleep 10
  done
  if [ "$arm" = W ]; then
    wake_held && res="$res
wakelock=held-at-end" || res="$res
wakelock=LOST"
    wake_release || true
  fi
  echo "$res" > "$OUT/trial-$n-$arm.txt"
  if [ $restarted = 1 ]; then
    log "trial $n [$arm] PHONE RESTARTED (before: $before)"
    sleep 60
    "$HERE/wedge_forensics.sh" post-reboot "doze-repro-trial$n" --bugreport
    break
  fi
  clean=$(echo "$res" | grep -c "device destroyed cleanly")
  if echo "$res" | grep -q "vkWaitForFences failed\|GPU driver"; then
    wedges=$((wedges + 1))
    log "trial $n [$arm] WEDGE (clean teardowns $clean/3; before: $before; after: $(state))"
    "$HERE/wedge_forensics.sh" event "doze-repro-trial$n-$arm" > "$OUT/forensics-$n.log" 2>&1
  elif echo "$res" | grep -q "^exit=0" && [ "$clean" = 3 ]; then
    times=$(echo "$res" | sed -n 's/.*time=\([0-9.]*\)ms.*/\1/p' | tr '\n' ' ')
    log "trial $n [$arm] clean (runs ms: $times; before: $before)"
  else
    log "trial $n [$arm] OTHER (see trial-$n-$arm.txt; before: $before)"
  fi
  [ $wedges -ge 3 ] && { log "3 wedges — stop"; break; }
done
log "done: $n trials, $wedges wedges (results: $OUT)"
