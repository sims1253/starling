package dev.starling.mobile.engine

/**
 * The speech models the on-device engine can load, keyed by the C API's
 * model kind (enum starling_ggml_model in cpp/include/starling_ggml.h). The
 * kind is never chosen by the user: [detect] reads it from the GGUF's own
 * metadata, at import (to refuse an unsupported file before it can replace
 * the active model) and again at load (to pick the engine).
 *
 * [workingSetBytes] is what a load needs on top of the weights (the file
 * size), for the memory gate. Both values are estimates, not measurements of
 * a particular phone: Parakeet's activations and graph buffers fit in 128 MiB
 * for the 0.6B class; the LLM-decoder kinds add a KV cache (up to ~450 MiB,
 * allocated on the first prefill) and larger compute buffers, so they ask
 * for 768 MiB.
 *
 * [maxWindowSeconds] is the longest audio one engine call may take. The
 * LLM-decoder kinds' encoders see at most 30 s (ARK's Whisper encoder caps
 * its mel at 3000 frames), so longer recordings are windowed below that;
 * Parakeet takes the serving layer's 30 s step plus 2 s of overlap.
 */
enum class ModelKind(
    val nativeKind: Int,
    val label: String,
    val workingSetBytes: Long,
    val maxWindowSeconds: Double = LLM_DECODER_WINDOW_SECONDS,
) {
    PARAKEET(
        1,
        "Parakeet",
        PARAKEET_WORKING_SET_BYTES,
        ChunkedTranscription.CHUNK_SECONDS + ChunkedTranscription.OVERLAP_SECONDS,
    ),
    MOSS(2, "MOSS-Transcribe", LLM_DECODER_WORKING_SET_BYTES),

    // ARK-ASR-0.6B rides the same engine (every dim comes from ark.* metadata),
    // so kind 3 loads both sizes; the C API's ARK06 alias is not needed.
    ARK(3, "ARK-ASR", LLM_DECODER_WORKING_SET_BYTES),
    GRANITE(6, "Granite Speech", LLM_DECODER_WORKING_SET_BYTES),

    // Likewise for Qwen3-ASR-0.6B and the QWEN3_06 alias.
    QWEN3(7, "Qwen3-ASR", LLM_DECODER_WORKING_SET_BYTES),
    VOXTRAL(11, "Voxtral Realtime", LLM_DECODER_WORKING_SET_BYTES),
    ;

    /** Free memory a load of a [modelBytes] file of this kind asks for (see [workingSetBytes]). */
    fun requiredMemoryBytes(modelBytes: Long): Long = modelBytes + workingSetBytes

    /** What [detect] found in a GGUF's metadata. */
    sealed interface Detection {
        /** Loadable as [kind]; [displayName] adds the size where the metadata tells it. */
        data class Supported(val kind: ModelKind, val displayName: String) : Detection

        /** Not a model the engine can load; [reason] is a short user-facing sentence. */
        data class Unsupported(val reason: String) : Detection
    }

    companion object {
        /** u32 metadata values [detect] reads to tell model sizes apart (retained by [GgufMetadata]). */
        const val ARK_HIDDEN_SIZE_KEY = "ark.llm.hidden_size"
        const val QWEN3_HIDDEN_SIZE_KEY = "qwen3.llm.hidden_size"

        private const val ARCHITECTURE_KEY = "general.architecture"

        /** Longest architecture tag echoed into the user-facing rejection message. */
        private const val MAX_ARCHITECTURE_DISPLAY_CHARS = 40

        /** general.architecture of every LLM-decoder kind (Parakeet is detected by its keys). */
        private val ARCHITECTURES = mapOf(
            "moss_transcribe" to MOSS,
            "ark" to ARK,
            "granite" to GRANITE,
            "qwen3" to QWEN3,
            "voxtral" to VOXTRAL,
        )

        /**
         * The model kind of a parsed GGUF. Parakeet comes first and by its
         * keys: every Parakeet dialect the engine loads (native parakeet.*,
         * parakeet.cpp/CrispASR, and transcribe.cpp stt.parakeet.* — see
         * cpp/parakeet/compat.cpp) carries metadata keys under "parakeet." or
         * "stt.parakeet.", while community files have no reliable
         * architecture string. Every other kind is named by
         * general.architecture, exactly as its native loader checks it
         * (cpp/lib/loader_kit.hpp check_gguf_header). An unrelated GGUF (a
         * llama-class chat model, a whisper model, the S1 text normalizer)
         * is unsupported.
         */
        fun detect(metadata: GgufMetadata.Parsed): Detection {
            if (metadata.keys.any { it.startsWith("parakeet.") || it.startsWith("stt.parakeet.") }) {
                return Detection.Supported(PARAKEET, PARAKEET.label)
            }
            val architecture = metadata.strings[ARCHITECTURE_KEY]
            val kind = ARCHITECTURES[architecture]
            if (kind != null) return Detection.Supported(kind, displayName(kind, metadata.uints))
            if (architecture == "s1") return Detection.Unsupported("S1-mini is a text model, not a speech model.")
            // Corruption-controlled up to the parse budget; cap what reaches
            // the user-facing message.
            val shown = architecture?.take(MAX_ARCHITECTURE_DISPLAY_CHARS) ?: "no architecture tag"
            return Detection.Unsupported("This GGUF ($shown) is not a supported speech model.")
        }

        /** [kind]'s label with its size, for the two kinds whose sizes share an engine. */
        private fun displayName(kind: ModelKind, uints: Map<String, Long>): String = when (kind) {
            ARK -> when (uints[ARK_HIDDEN_SIZE_KEY]) {
                2048L -> "${kind.label} 3B"
                896L -> "${kind.label} 0.6B"
                else -> kind.label
            }
            QWEN3 -> when (uints[QWEN3_HIDDEN_SIZE_KEY]) {
                2048L -> "${kind.label} 1.7B"
                1024L -> "${kind.label} 0.6B"
                else -> kind.label
            }
            else -> kind.label
        }
    }
}

// Top-level so the enum entries can use them (a companion is initialized after its entries).
private const val PARAKEET_WORKING_SET_BYTES = 128L * 1024 * 1024
private const val LLM_DECODER_WORKING_SET_BYTES = 768L * 1024 * 1024
private const val LLM_DECODER_WINDOW_SECONDS = 30.0
