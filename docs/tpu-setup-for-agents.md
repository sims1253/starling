# Pixel TPU encoder: let your coding agent set it up

On a Pixel with a Tensor G5 (Pixel 10 series), Starling can run the speech
encoder of Parakeet-TDT models (Parakeet, the Bark 3.0 family) and of
Qwen3-ASR on the phone's TPU instead of its GPU: on a Pixel 10 Pro a 32 s
window of Parakeet takes 0.47 s instead of 1.94 s and about a fifth of the
energy, with the same transcripts. How it works and how it was measured:
[fast engines, "Pixel TPU encoder"](fast-engine.md).

Nothing for the TPU ships with the app. Each phone compiles its own files
with its own on-device compiler, over adb from a computer, for the exact
model files the app has loaded. That is a developer workflow: an
afternoon of tooling for a person, a single prompt for a coding agent
(Claude Code, Codex, and so on) running on your computer. This page is that
prompt plus the few steps only you can do.

## Before you start (you, not the agent)

1. **Back up what matters on the phone.** The phone's compiler can run it out
   of memory and reboot it. Starling only compiles graphs that have compiled
   cleanly before and refuses with less than 4 GB free, but a reboot is still
   possible.
2. **Enable Wireless debugging** (Settings → System → Developer options) and
   keep the phone on the same network as the computer. Pairing needs the
   code the phone shows, so the agent will ask you to read it out or pair
   yourself.
3. **Load every model you use in the app once.** The app then records which
   files it would use on the TPU, and the agent builds exactly those.
4. **Accept the Android SDK license yourself.** The agent needs Google's
   Android NDK and platform-tools, which come under the Android Software
   Development Kit License Agreement. The prompt tells the agent to stop and
   let you run the acceptance step.

## The prompt

Paste this into your agent, in a checkout of
<https://github.com/sims1253/starling>:

```text
Set up the Pixel TPU encoder for the Starling app on my phone, following
docs/tpu-setup-for-agents.md and docs/fast-engine.md ("Pixel TPU encoder").

Rules:
- Only touch my phone through adb, only under /data/local/tmp/starling-npu
  and the Starling app's own npu directories, as scripts/npu/*.sh do. Never
  reboot it, change its settings, or uninstall anything.
- Hold /tmp/starling-phone.lock around phone work (the scripts do).
- Before installing the Android NDK or platform-tools, stop and ask me to
  accept the Android SDK license myself (sdkmanager --licenses); do not
  accept it for me or pipe "yes" into it.
- If a compile fails or the phone disconnects, stop and tell me; do not retry
  in a loop.

Steps:
1. git submodule update --init --recursive.
2. Tools: adb (platform-tools), CMake >= 3.22 and Ninja, the Android NDK
   28.2.13676358, and a Python 3.10-3.13 virtualenv with gguf, torch (CPU
   build is enough), litert-torch and ai-edge-quantizer (xxhash optional).
3. Connect to the phone over wireless debugging (adb pair / adb connect; ask
   me for the pairing code) and check `adb devices`.
4. Cross-build the phone tool:
     cmake -S . -B build-android -G Ninja \
       -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
       -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-30 \
       -DCMAKE_BUILD_TYPE=Release -DSTARLING_FAST=ON -DGGML_NATIVE=OFF \
       -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF \
       -DSTARLING_GGML_TESTS=OFF -DSTARLING_QUANTIZE=OFF
     cmake --build build-android --target starling-npu-tool
5. Run, with PY pointing at the virtualenv's python:
     NPU_TOOL=build-android/starling-npu-tool scripts/npu/npu_compile_all.sh --dry-run
   It lists the models the app asked for. The host needs each model's GGUF,
   the very file the app downloaded: if the script cannot pull it from the
   phone, download the same file from the app's catalog
   (apps/mobile/app/src/main/java/dev/starling/mobile/engine/ModelCatalog.kt,
   pinned URL and SHA-256) and pass it with --gguf.
6. Run the same command without --dry-run, then tell me to reopen the app.
   Each model's status should read ok (adb logcat -s StarlingNpu).
7. Report what you built, for which models, and anything that failed.
```

After an Android system update the TPU firmware changes and the app reports
the files as stale; run the same prompt again.

## Licenses

Everything this installs or builds stays on your computer and your phone:
Starling distributes no TPU files and no Google software.

- The scripts and the engine code they build are Starling's: MIT, and
  Apache-2.0 for `cpp/npu`, whose Pixel TPU client is partly derived from
  Google's Apache-2.0 [LiteRT](https://github.com/google-ai-edge/LiteRT).
- The compiler and TPU runtime are the phone's own vendor library
  (`libedgetpu_litert.so`), which the engine loads at run time; Starling
  neither ships nor modifies it. On the phone, its compile service only
  serves apps on Google's allowlist, which is why compiling runs from the
  adb shell. We know of no published terms for that library.
- The Android NDK and platform-tools come from Google under the Android SDK
  License Agreement, which you accept when you install them; torch,
  litert-torch and ai-edge-quantizer under their own (BSD-style and
  Apache-2.0) licenses.
- The compiled files are built from the model files on your phone and never
  leave your devices, so the models' licenses (CC-BY-4.0, Apache-2.0) ask
  nothing further.
