#!/usr/bin/env bash
# phone_bench_ab.sh — generic trusted phone A/B gate (issue #176): the
# phone_gates.sh protocol generalized to ANY model. Pushes BASE_BIN and
# CAND_BIN (host paths), then runs ROUNDS alternating base/cand invocations
# (order alternates each round, screen off, `timeout` on the remote side)
# and prints METRIC lines:
#
#   METRIC total_ms=…            median candidate total ms (per timed run)
#   METRIC total_ms_base=…       median baseline total ms
#   METRIC total_ms_delta_pct=…  (cand-base)/base*100  (negative = faster)
#   METRIC total_ms_min/max, total_ms_base_min/max
#   METRIC transcripts_match=1|0 (empty transcripts -> 0 + exit 3)
#   METRIC peak_rss_kb=…         best-effort VmHWM sample of the remote cand
#                                process (unavailable when not sampleable)
#
# Options (env equivalents in capitals): --base-bin --cand-bin --model
# --gguf --wav --engine --rounds --runs --dev --threads --extra-env-cand
# --extra-env-base. GGUF/WAV are host paths, pushed to $DEV when missing.
#
# Sources {trusted}/benchmarks/fast_engine/phone_common.sh (kill_benches /
# wait_benches / screen_off); the trap kills remote benches on any exit.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)                 # <trusted>/benchmarks/campaign/gates
TRUSTED=${TRUSTED:-$(cd "$HERE/../../.." && pwd)}
# shellcheck source=benchmarks/fast_engine/phone_common.sh
. "$TRUSTED/benchmarks/fast_engine/phone_common.sh"

DEV=${DEV:-/data/local/tmp/starling}
MODEL=${MODEL:-parakeet}
GGUF=${GGUF:-}
WAV=${WAV:-}
ENGINE=${ENGINE:-fast}
ROUNDS=${ROUNDS:-3}
RUNS=${RUNS:-4}
THREADS=${THREADS:-6}
EXTRA_ENV_BASE=${EXTRA_ENV_BASE:-}
EXTRA_ENV_CAND=${EXTRA_ENV_CAND:-}
BASE_BIN=${BASE_BIN:-}
CAND_BIN=${CAND_BIN:-}

# Values interpolated into the remote `adb shell` command string must be
# single safe tokens: anything with spaces or shell metacharacters would
# split or execute on the device instead of failing loudly here.
safe_token() { # non-empty, [A-Za-z0-9_=:./%+-] only
  case "$1" in
    ''|*[!A-Za-z0-9_=:./%+-]*) return 1 ;;
  esac
  return 0
}
positive_int() { case "$1" in ''|*[!0-9]*|0) return 1 ;; esac; return 0; }
safe_env_list() { # "NAME=value [NAME=value ...]"; values must be safe tokens
  [ -n "$1" ] || return 0
  local w
  for w in $1; do # shellcheck disable=SC2086 — word splitting is the point
    case "$w" in
      [A-Za-z_][A-Za-z0-9_]*=*) ;;
      *) return 1 ;;
    esac
    safe_token "${w#*=}" || return 1
  done
  return 0
}

while [ $# -gt 0 ]; do
  case "$1" in
    --base-bin) BASE_BIN=$2; shift 2 ;;
    --cand-bin) CAND_BIN=$2; shift 2 ;;
    --model) MODEL=$2; shift 2 ;;
    --gguf) GGUF=$2; shift 2 ;;
    --wav) WAV=$2; shift 2 ;;
    --engine) ENGINE=$2; shift 2 ;;
    --rounds) ROUNDS=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --dev) DEV=$2; shift 2 ;;
    --threads) THREADS=$2; shift 2 ;;
    --extra-env-base) EXTRA_ENV_BASE=$2; shift 2 ;;
    --extra-env-cand) EXTRA_ENV_CAND=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -n "$BASE_BIN" ] && [ -n "$CAND_BIN" ] && [ -n "$GGUF" ] && [ -n "$WAV" ] || {
  echo "need --base-bin, --cand-bin, --gguf, --wav" >&2; exit 2; }
[ -f "$BASE_BIN" ] || { echo "missing base binary: $BASE_BIN" >&2; exit 2; }
[ -f "$CAND_BIN" ] || { echo "missing cand binary: $CAND_BIN" >&2; exit 2; }
[ -f "$GGUF" ] || { echo "missing gguf: $GGUF" >&2; exit 2; }
[ -f "$WAV" ] || { echo "missing wav: $WAV" >&2; exit 2; }

command -v adb >/dev/null || { echo "adb missing" >&2; exit 2; }
adb get-state >/dev/null 2>&1 || { echo "phone not connected" >&2; exit 1; }

GGUF_NAME=$(basename "$GGUF"); WAV_NAME=$(basename "$WAV")
for v in "$ROUNDS" "$RUNS" "$THREADS"; do
  positive_int "$v" || { echo "rounds/runs/threads must be positive integers: $v" >&2; exit 2; }
done
for v in "$DEV" "$MODEL" "$ENGINE" "$GGUF_NAME" "$WAV_NAME"; do
  safe_token "$v" || { echo "value has spaces/metacharacters (unsafe remotely): $v" >&2; exit 2; }
done
safe_env_list "$EXTRA_ENV_BASE" || {
  echo "--extra-env-base must be NAME=value tokens without spaces/metacharacters" >&2; exit 2; }
safe_env_list "$EXTRA_ENV_CAND" || {
  echo "--extra-env-cand must be NAME=value tokens without spaces/metacharacters" >&2; exit 2; }
adb shell "mkdir -p $DEV" >/dev/null 2>&1 || true
adb push "$BASE_BIN" "$DEV/starling-bench-base" >/dev/null
adb push "$CAND_BIN" "$DEV/starling-bench-cand" >/dev/null
have() { adb shell "[ -f $DEV/$1 ]" >/dev/null 2>&1; }
have "$GGUF_NAME" || adb push "$GGUF" "$DEV/$GGUF_NAME" >/dev/null
have "$WAV_NAME" || adb push "$WAV" "$DEV/$WAV_NAME" >/dev/null

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"; kill_benches' EXIT

bench() { # <remote-binary> <extra-env> <out-file>
  wait_benches
  adb shell "cd $DEV && timeout 600 env LD_LIBRARY_PATH=. STARLING_ENGINE=$ENGINE \
    STARLING_GGML_THREADS=$THREADS STARLING_FAST_CACHE_DIR=$DEV STARLING_FAST_TIMING=1 $2 \
    ./$1 --model $MODEL --gguf $DEV/$GGUF_NAME --warmup --runs $RUNS $DEV/$WAV_NAME" \
    > "$3" 2>&1
}

median() { sort -n | awk '{a[NR]=$1} END {if (!NR) exit 1; print (NR % 2) ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2}'; }
total_ms() { sed -n 's/.*time=\([0-9.]*\)ms.*/\1/p' "$1"; }
last_tr() { sed -n 's/^  //p' "$1" | tail -1; }

# best-effort VmHWM sampling of the remote candidate process while it runs
sample_rss() { # echoes max kb seen so far (or nothing)
  local pid out
  # comm is truncated to 15 chars: "pidof starling-bench-cand" never matches.
  # The truncated name covers both pushed binaries, but this only samples
  # while the candidate alone runs (wait_benches ran beforehand).
  pid=$(adb shell "pidof starling-bench- 2>/dev/null" 2>/dev/null | tr -d '\r' | awk '{print $1}')
  [ -n "$pid" ] || return 0
  out=$(adb shell "cat /proc/$pid/status 2>/dev/null" 2>/dev/null | tr -d '\r' | sed -n 's/^VmHWM:[[:space:]]*\([0-9]*\).*/\1/p')
  [ -n "$out" ] && echo "$out"
}

bench_with_rss() { # <remote-binary> <extra-env> <out-file>
  bench "$1" "$2" "$3" &
  local bpid=$! rss max=0 v rc=0
  while kill -0 "$bpid" 2>/dev/null; do
    v=$(sample_rss || true)
    case "$v" in ''|*[!0-9]*) : ;; *) [ "$v" -gt "$max" ] && max=$v ;; esac
    sleep 0.3
  done
  # Propagate the bench's exit status. Writing the rss sample afterwards must
  # not mask a failed bench: callers decide fail/inconclusive from this status
  # (the `||` at the call sites keeps set -e off inside this function).
  wait "$bpid"; rc=$?
  echo "$max" > "$3.rss"
  return "$rc"
}

kill_benches
screen_off
# A bench that exited 0 must still have produced --runs timings: a truncated
# run (hung run killed by the remote timeout, skipped transcription) would
# otherwise yield a fast-biased median over the surviving subset and a wrong
# pass. Fail loudly instead.
require_complete() { # <out-file> <label>
  local got
  got=$(total_ms "$1" | awk 'END {print NR}')
  if [ "$got" != "$RUNS" ]; then
    echo "ERROR: $2 bench output truncated: $got/$RUNS time= lines" >&2
    tail -5 "$1" >&2
    exit 1
  fi
}
declare -a base_vals cand_vals
for r in $(seq 1 "$ROUNDS"); do
  screen_off
  # Alternate the order each round (the second side runs on a warmer GPU).
  if [ $((r % 2)) = 1 ]; then
    bench_with_rss starling-bench-base "$EXTRA_ENV_BASE" "$TMP/b$r" || { echo "base bench failed (round $r)" >&2; tail -5 "$TMP/b$r" >&2; exit 1; }
    bench_with_rss starling-bench-cand "$EXTRA_ENV_CAND" "$TMP/c$r" || { echo "cand bench failed (round $r)" >&2; tail -5 "$TMP/c$r" >&2; exit 1; }
  else
    bench_with_rss starling-bench-cand "$EXTRA_ENV_CAND" "$TMP/c$r" || { echo "cand bench failed (round $r)" >&2; tail -5 "$TMP/c$r" >&2; exit 1; }
    bench_with_rss starling-bench-base "$EXTRA_ENV_BASE" "$TMP/b$r" || { echo "base bench failed (round $r)" >&2; tail -5 "$TMP/b$r" >&2; exit 1; }
  fi
  require_complete "$TMP/b$r" base
  require_complete "$TMP/c$r" cand
  base_vals+=("$(total_ms "$TMP/b$r" | median)")
  cand_vals+=("$(total_ms "$TMP/c$r" | median)")
  sleep 5   # breathe between rounds
done

for v in "${base_vals[@]}" "${cand_vals[@]}"; do
  # one decimal point at most: "1.2.3" passes a plain char-class check and
  # then breaks the awk delta expression with a syntax error
  if ! [[ "$v" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "ERROR: non-numeric total_ms metric — bench output unparsed: '$v'" >&2; exit 1
  fi
done
base=$(printf '%s\n' "${base_vals[@]}" | median)
cand=$(printf '%s\n' "${cand_vals[@]}" | median)
all_base=$(printf '%s\n' "${base_vals[@]}")
all_cand=$(printf '%s\n' "${cand_vals[@]}")
awk "BEGIN {exit !($base > 0)}" || { echo "ERROR: base metric is zero" >&2; exit 1; }
delta=$(awk "BEGIN {printf \"%.2f\", ($cand - $base) / $base * 100}")

# Transcript check from the last round's bench outputs (no extra loads).
ta=$(last_tr "$TMP/b$ROUNDS"); tb=$(last_tr "$TMP/c$ROUNDS")
MATCH=1
if [ -z "$ta" ] || [ -z "$tb" ]; then
  echo "transcripts empty on at least one side; cannot confirm equality" >&2
  MATCH=0
  INCONCLUSIVE=1
elif [ "$ta" != "$tb" ]; then
  MATCH=0
  echo "DIVERGENCE: transcripts differ between base and cand (last round)" >&2
fi

PEAK=$(cat "$TMP/c$ROUNDS.rss" 2>/dev/null || echo 0)
case "$PEAK" in ''|*[!0-9]*) PEAK=0 ;; esac

echo "rounds(base): ${base_vals[*]}"
echo "rounds(cand): ${cand_vals[*]}"
echo "METRIC total_ms=$cand"
echo "METRIC total_ms_base=$base"
echo "METRIC total_ms_delta_pct=$delta"
echo "METRIC total_ms_min=$(printf '%s\n' "$all_cand" | sort -n | head -1)"
echo "METRIC total_ms_max=$(printf '%s\n' "$all_cand" | sort -n | tail -1)"
echo "METRIC total_ms_base_min=$(printf '%s\n' "$all_base" | sort -n | head -1)"
echo "METRIC total_ms_base_max=$(printf '%s\n' "$all_base" | sort -n | tail -1)"
echo "METRIC transcripts_match=$MATCH"
if [ "$PEAK" -gt 0 ] 2>/dev/null; then
  echo "METRIC peak_rss_kb=$PEAK"
else
  echo "METRIC peak_rss_kb=unavailable"
fi
[ -n "${INCONCLUSIVE:-}" ] && exit 3
exit 0
