package dev.starling.mobile.engine

/**
 * A model the app can fetch itself. The URL pins a repository revision, so
 * the bytes behind it never change; [sizeBytes] and [sha256] are checked
 * before the file can replace the active model.
 */
data class ModelDownload(
    val url: String,
    val sizeBytes: Long,
    val sha256: String,
    /** Shown in the model list instead of the long file name. */
    val label: String,
    /** What the file loads as; import validation detects the same kind from its metadata. */
    val kind: ModelKind,
) {
    /** Installed under the file name it has upstream, so it is recognizable in the model list. */
    val fileName: String = url.substringAfterLast('/')

    init {
        require(fileName.endsWith(".gguf")) { "A catalog model URL must end in a .gguf file name: $url" }
    }
}

object ModelCatalog {
    /**
     * The recommended on-device model: Parakeet TDT 0.6B v3, q4_k_m with F16
     * convolutions (553 MB), the smallest q4-class file of the published
     * set; runs on the Vulkan fast engine where available.
     */
    val RECOMMENDED_PARAKEET = ModelDownload(
        url = "https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf/resolve/" +
            "96402b32bd374742aa1da3c66af30aa64cea3fdb/parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf",
        sizeBytes = 552_670_624L,
        sha256 = "2b5ea37e3193c71b3ad2f859b4238ff4501898ba73d63899568faee1daae9982",
        label = "Parakeet TDT 0.6B v3",
        kind = ModelKind.PARAKEET,
    )

    /**
     * Every catalog model, the recommended one first; their file names are
     * reserved for verified downloads (an import never takes one).
     *
     * Every other [ModelKind] loads today through Import model (.gguf).
     * Per-model phone quants (MOSS-Transcribe, ARK-ASR, Granite Speech,
     * Qwen3-ASR, Voxtral Realtime) are added here once they are hosted at a
     * pinned revision with a known size and SHA-256 — never before: an entry
     * without verified bytes would let a download replace the active model
     * with unchecked data.
     */
    val ALL = listOf(
        RECOMMENDED_PARAKEET,
        // Per-model phone quants go here once published (see above).
    )

    /** The catalog entry installed as [fileName], if any. */
    fun forFileName(fileName: String): ModelDownload? = ALL.firstOrNull { it.fileName == fileName }
}
