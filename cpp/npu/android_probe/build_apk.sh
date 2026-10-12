#!/bin/bash
# Builds the test-only NPU probe APK with the SDK command-line tools (no Gradle).
# Needs cpp/npu/build_android.sh to have produced libstarling_npu_probe.so.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SDK=${ANDROID_HOME:-$HOME/android-sdk}
BT=$(ls -d "$SDK"/build-tools/* | sort -V | tail -1)
JAR=$(ls -d "$SDK"/platforms/android-*/android.jar | sort -V | tail -1)
NPU_OUT=${NPU_OUT:-$HERE/../../../build-npu}
OUT=${OUT:-$NPU_OUT/apk}
rm -rf "$OUT"; mkdir -p "$OUT/classes" "$OUT/dex" "$OUT/stage/lib/arm64-v8a"
"$BT/aapt2" link -o "$OUT/base.apk" --manifest "$HERE/AndroidManifest.xml" -I "$JAR"
javac --release 11 -cp "$JAR" -d "$OUT/classes" $(find "$HERE/src" -name '*.java')
"$BT/d8" --min-api 31 --lib "$JAR" --output "$OUT/dex" $(find "$OUT/classes" -name '*.class')
cp "$OUT/dex/classes.dex" "$OUT/stage/"
cp "$NPU_OUT/libstarling_npu_probe.so" "$OUT/stage/lib/arm64-v8a/"
cp "$OUT/base.apk" "$OUT/unsigned.apk"
(cd "$OUT/stage" && zip -q -r "$OUT/unsigned.apk" classes.dex lib)
"$BT/zipalign" -f -p 4 "$OUT/unsigned.apk" "$OUT/aligned.apk"
"$BT/apksigner" sign --ks "$HOME/.android/debug.keystore" --ks-pass pass:android --key-pass pass:android \
  --out "$OUT/npuprobe.apk" "$OUT/aligned.apk"
ls -la "$OUT/npuprobe.apk"
