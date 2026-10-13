#!/bin/bash
# Retry phone_session.sh (under the phone lock) every 3 minutes while the
# user has an app in front (exit 3), up to ~2 hours.
#   benchmarks/litert_gpu/retry_session.sh LOGDIR PHASE...
HERE=$(cd "$(dirname "$0")" && pwd)
for i in $(seq 1 40); do
  flock -w 3600 /tmp/starling-phone.lock "$HERE/${SESSION:-phone_session.sh}" "$@"
  rc=$?
  [ $rc -ne 3 ] && exit $rc
  echo "phone in use, retry $i in 180 s ($(date +%T))"
  sleep 180
done
exit 3
