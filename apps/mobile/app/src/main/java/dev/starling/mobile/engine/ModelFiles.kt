package dev.starling.mobile.engine

import java.io.File

/** Pure checks for a candidate on-device model file before it is accepted. */
object ModelFiles {
    /** A 0.6B-class q4 GGUF is a few hundred MB; anything smaller is suspect. */
    const val MIN_MODEL_BYTES = 50L * 1024 * 1024

    const val GGUF_MAGIC_SIZE = 4

    private const val TOO_SMALL = "This file is too small to be a speech model."

    /**
     * Null when [sizeBytes] and the first [header.size] bytes look like a
     * GGUF the engine can attempt; otherwise a short user-facing reason.
     * The shallow header-sniff contract for callers that already hold the
     * first bytes; the staged-import path uses [validateStaged]'s deep
     * single-open check instead.
     */
    fun validate(sizeBytes: Long, header: ByteArray): String? {
        if (sizeBytes < MIN_MODEL_BYTES) {
            return TOO_SMALL
        }
        val magic = "GGUF".toByteArray(Charsets.US_ASCII)
        if (header.size < magic.size || !header.copyOfRange(0, magic.size).contentEquals(magic)) {
            return "This file is not a GGUF model."
        }
        return null
    }

    /**
     * Deep check of a model [file]: size, a bounded parse of the GGUF
     * metadata and tensor-info sections, and the model kind
     * ([ModelKind.detect]). One open: [GgufMetadata.parse] reads and checks
     * the magic and version itself, so no separate header pre-read (two
     * opens could observe different file states). Runs before promotion
     * ([validateStaged]) and again at each load, where the kind picks the
     * engine.
     *
     * Intentionally conservative divergence from the native loader: a
     * hypothetical hand-built model without the metadata [ModelKind.detect]
     * reads, but with loadable tensor names, is rejected at import rather
     * than after replacing the last usable model.
     */
    fun inspect(file: File): ModelKind.Detection {
        if (file.length() < MIN_MODEL_BYTES) {
            return ModelKind.Detection.Unsupported(TOO_SMALL)
        }
        val metadata = GgufMetadata.parse(file)
            ?: return ModelKind.Detection.Unsupported("This model file is corrupt or truncated.")
        return ModelKind.detect(metadata)
    }

    /**
     * Pre-promotion check of a fully staged model [file] (B08): null when it
     * may be promoted; otherwise [inspect]'s short user-facing reason. An
     * unrelated GGUF (a llama-class chat model, a whisper model, the S1 text
     * model, ...) is rejected here, before it can replace the active model.
     */
    fun validateStaged(file: File): String? =
        (inspect(file) as? ModelKind.Detection.Unsupported)?.reason
}
