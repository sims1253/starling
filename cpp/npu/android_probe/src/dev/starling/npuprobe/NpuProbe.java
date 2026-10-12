// Copyright 2026 The Starling Authors. Apache-2.0.
package dev.starling.npuprobe;

/** JNI bridge to cpp/npu (libstarling_npu_probe.so). */
public final class NpuProbe {
    static {
        System.loadLibrary("starling_npu_probe");
    }

    private NpuProbe() {}

    public static native String compile(String tflite, String artifact, String function, long[] inBytes,
                                        long[] outBytes);

    public static native String run(String artifact, String in0, String in1, String outPrefix, int runs);
}
