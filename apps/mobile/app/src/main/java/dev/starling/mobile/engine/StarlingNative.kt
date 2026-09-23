package dev.starling.mobile.engine

/**
 * JNI surface of libstarling_jni, the on-device build of the repository's
 * native engine. Handles are engine contexts; [load] returns 0 on failure
 * and [lastError] then carries the reason. Calls are serialized inside the
 * engine, but callers should still drive them from one thread.
 */
object StarlingNative {
    /** ABI the JNI library is built against; refused on mismatch. */
    const val EXPECTED_ABI_VERSION = 6

    init {
        System.loadLibrary("starling_jni")
    }

    external fun abiVersion(): Int

    /**
     * The ggml device the engine runs on ("CPU", "Vulkan0", ...) once a model
     * has loaded; before that, the compiled backend family.
     */
    external fun backendName(): String?

    /** Loads STARLING_GGML_PARAKEET_TDT from [ggufPath]; 0 on failure. */
    external fun load(ggufPath: String): Long

    /** Mono float32 samples in [-1, 1] at [sampleRate] Hz; null on failure. */
    external fun transcribe(handle: Long, samples: FloatArray, sampleRate: Int): String?

    external fun free(handle: Long)

    external fun lastError(handle: Long): String?
}
