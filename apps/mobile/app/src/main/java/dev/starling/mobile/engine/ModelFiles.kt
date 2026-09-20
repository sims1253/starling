package dev.starling.mobile.engine

import java.io.File

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

    /**
     * Deep pre-promotion check of a fully staged model [file] (B08): size,
     * magic, a bounded parse of the GGUF metadata section, and the model
     * family. Null when the file may be promoted; otherwise a short
     * user-facing reason.
     *
     * Family rule: every GGUF dialect the native engine loads (native
     * parakeet.*, parakeet.cpp/CrispASR, and transcribe.cpp stt.parakeet.* —
     * see cpp/parakeet/compat.cpp) carries metadata keys under "parakeet."
     * or "stt.parakeet.". An unrelated GGUF (a llama-class chat model, a
     * whisper model, ...) carries none, and is rejected here before it can
     * replace the active model. Intentionally conservative divergence from
     * the native loader: a hypothetical hand-built model with neither key
     * family but loadable tensor names would be rejected at import rather
     * than after replacing the last usable model.
     */
    fun validateStaged(file: File): String? {
        val header = runCatching {
            java.io.DataInputStream(file.inputStream().buffered()).use { stream ->
                // A single read may legally return short; readFully cannot.
                ByteArray(GGUF_MAGIC_SIZE).also { stream.readFully(it) }
            }
        }.getOrNull() ?: return "This model file could not be read."
        val shallow = validate(file.length(), header)
        if (shallow != null) return shallow
        val metadata = GgufMetadata.parse(file) ?: return "This model file is corrupt or truncated."
        val parakeetFamily = metadata.keys.any { it.startsWith("parakeet.") || it.startsWith("stt.parakeet.") }
        if (!parakeetFamily) {
            val architecture = metadata.strings["general.architecture"]
            return "This GGUF (${architecture ?: "no architecture tag"}) is not a Parakeet model."
        }
        return null
    }
}
