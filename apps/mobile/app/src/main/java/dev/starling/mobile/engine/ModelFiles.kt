package dev.starling.mobile.engine

/** Pure checks for a candidate on-device model file before it is accepted. */
object ModelFiles {
    /** A 0.6B-class q4 GGUF is a few hundred MB; anything smaller is suspect. */
    const val MIN_MODEL_BYTES = 50L * 1024 * 1024

    const val GGUF_MAGIC_SIZE = 4

    /**
     * Null when [sizeBytes] and the first [header.size] bytes look like a
     * GGUF the engine can attempt; otherwise a short user-facing reason.
     */
    fun validate(sizeBytes: Long, header: ByteArray): String? {
        if (sizeBytes < MIN_MODEL_BYTES) {
            return "This file is too small to be a Parakeet model."
        }
        val magic = "GGUF".toByteArray(Charsets.US_ASCII)
        if (header.size < magic.size || !header.copyOfRange(0, magic.size).contentEquals(magic)) {
            return "This file is not a GGUF model."
        }
        return null
    }
}
