#!/usr/bin/env bash
# build.sh — compile WakeHold.java to wakehold.dex (needs a JDK and the
# Android SDK: ANDROID_HOME, a platform android.jar and build-tools' d8).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SDK=${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Android/Sdk}}
JAR=$(ls -d "$SDK"/platforms/android-*/android.jar | sort -V | tail -1)
D8=$(ls -d "$SDK"/build-tools/*/d8 | sort -V | tail -1)
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
javac --release 11 -cp "$JAR" -d "$TMP" "$HERE/WakeHold.java"
"$D8" --min-api 26 --lib "$JAR" --output "$TMP" "$TMP"/*.class
mv "$TMP/classes.dex" "$HERE/wakehold.dex"
echo "$HERE/wakehold.dex"
