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

# SIGKILL any bench left running: a locally timed-out adb shell leaves the
# remote bench alive (its output pipe is gone; it can hang in poll forever).
kill_benches() {
  adb shell "for p in \$(pidof $BENCH_BINS); do kill -9 \$p; done" >/dev/null 2>&1 || true
}

# Back-to-back model loads (1.6 GB each) need the previous process gone:
# wait up to 120 s for it, then kill it.
wait_benches() {
  timeout 120 adb shell "while pidof $BENCH_BINS >/dev/null 2>&1; do sleep 1; done" \
    >/dev/null 2>&1 || kill_benches
}
