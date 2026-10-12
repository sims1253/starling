#!/bin/bash
# Quick phone probe: OpenCL libraries visible to apps, display state, free space.
# Run under: flock /tmp/starling-phone.lock benchmarks/litert_gpu/probe.sh
set -u
export ANDROID_SERIAL=${ANDROID_SERIAL:-192.168.178.59:45929}
adb shell 'ls -la /vendor/lib64/libOpenCL* 2>&1; grep -i opencl /vendor/etc/public.libraries.txt; getprop ro.hardware.vulkan; getprop ro.board.platform; dumpsys power | grep -m1 -o "mWakefulness=[A-Za-z]*"; ls /data/local/tmp/; df -h /data | tail -1; pidof starling-bench npu_bench litert_gpu_bench'
