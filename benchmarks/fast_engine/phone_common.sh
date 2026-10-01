# phone_common.sh — helpers shared by the phone measurement scripts
# (phone_gates.sh, phone_energy.sh). Sourced, not run.

BENCH_BINS="starling-bench starling-bench-base starling-bench-cand"

# Current display state, e.g. "mWakefulness=Asleep" ("" when unreadable).
wakefulness() {   # never fails: an unreadable state prints ""
  { adb shell "dumpsys power | grep -m1 -o 'mWakefulness=[A-Za-z]*'" 2>/dev/null || true; } | tr -d '\r'
}

# Put the display to sleep and verify it: with the screen awake the
# compositor shares the GPU and decode times swing 2x (measured), so a
# screen that stays awake invalidates the measurement.
screen_off() {
  local state
  state=$(wakefulness)
  if [ "$state" = "mWakefulness=Awake" ]; then
    adb shell "input keyevent KEYCODE_POWER" >/dev/null
    sleep 2
    state=$(wakefulness)
  fi
  case "$state" in
    mWakefulness=Awake) echo "ERROR: display still awake after KEYCODE_POWER" >&2; return 1 ;;
    mWakefulness=*) return 0 ;;
    *) echo "ERROR: could not read the display state (got '$state')" >&2; return 1 ;;
  esac
}

# Wait up to $1 s (bounded on the device and on the host) for every bench
# to exit; succeeds only if none is left.
benches_gone() {
  timeout $(( $1 + 10 )) adb shell "i=0; while pidof $BENCH_BINS >/dev/null 2>&1 && [ \$i -lt $1 ]; do sleep 1; i=\$((i+1)); done; ! pidof $BENCH_BINS >/dev/null 2>&1" \
    >/dev/null 2>&1
}

# Stop any bench left running: a locally timed-out adb shell leaves the
# remote bench alive (its output pipe is gone; it can hang in poll forever).
# #325 hygiene: SIGTERM first and give the process up to 10 s to exit through
# its destructors (SIGKILL abandons a live VkDevice to the driver's async
# reaping — a leading wedge correlate, RESEARCH_LOG P2-7). MOSS stops at the
# next decode round; other engines finish the current call (seconds) and
# stop between calls. KILL is the fallback, not the default. Every adb call
# is bounded on the host, and the wait loop on the device too, so a host
# timeout never orphans a remote shell. Fails if a bench survives KILL (the
# next 1.6 GB load must not collide with it); in EXIT traps call it as
# `kill_benches || true` so cleanup cannot mask the script's own status.
kill_benches() {
  timeout 15 adb shell "for p in \$(pidof $BENCH_BINS); do kill -TERM \$p; done" >/dev/null 2>&1 || true
  benches_gone 10 && return 0
  timeout 15 adb shell "for p in \$(pidof $BENCH_BINS); do kill -9 \$p; done" >/dev/null 2>&1 || true
  benches_gone 10 && return 0
  echo "WARNING: a bench is still alive (or adb is unreachable) after SIGKILL" >&2
  return 1
}

# Back-to-back model loads (1.6 GB each) need the previous process gone:
# wait up to 120 s for it, then kill it (bounded on the device as well).
wait_benches() {
  benches_gone 120 || kill_benches
}
