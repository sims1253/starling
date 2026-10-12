package dev.starling.mobile.engine

/**
 * JNI surface of libstarling_jni, the on-device build of the repository's
 * native engine. Handles are engine contexts; [load] returns 0 on failure
 * and [lastError] then carries the reason. Calls are serialized inside the
 * engine, but callers should still drive them from one thread.
 */
object StarlingNative {
    /** ABI the JNI library is built against; refused on mismatch. */
    const val EXPECTED_ABI_VERSION = 10

    init {
        System.loadLibrary("starling_jni")
    }

    external fun abiVersion(): Int

    /**
     * Loads [ggufPath] as the engine [kind] (a starling_ggml_model value,
     * [ModelKind.nativeKind]); 0 on failure, including a kind the app does
     * not support.
     */
    external fun load(ggufPath: String, kind: Int): Long

    /** Mono float32 samples in [-1, 1] at [sampleRate] Hz; null on failure. */
    external fun transcribe(handle: Long, samples: FloatArray, sampleRate: Int): String?

    /** Polled by the engine during [transcribeCancellable], on the calling thread. */
    fun interface Cancel {
        /** True asks the call in progress to stop. */
        fun requested(): Boolean
    }

    /**
     * [transcribe] that stops at the engine's next checkpoint (between
     * pipeline stages and decoder steps; not inside the GPU encoder
     * submission) once [cancel] answers true. A stopped call returns null
     * with [lastError] [CANCELLED_ERROR].
     */
    external fun transcribeCancellable(handle: Long, samples: FloatArray, sampleRate: Int, cancel: Cancel): String?

    /** [lastError] of a call [transcribeCancellable] stopped (cpp/runtime/call_abort.hpp). */
    const val CANCELLED_ERROR = "cancelled"

    external fun free(handle: Long)

    external fun lastError(handle: Long): String?
}
