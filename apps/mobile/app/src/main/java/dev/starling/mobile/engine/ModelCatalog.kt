package dev.starling.mobile.engine

/**
 * A model the app can fetch itself. The URL pins a repository revision, so
 * the bytes behind it never change; [sizeBytes] and [sha256] are checked
 * before the file can replace the active model.
 */
data class ModelDownload(
    val displayName: String,
    val url: String,
    val sizeBytes: Long,
    val sha256: String,
)

object ModelCatalog {
    /**
     * The recommended on-device model: Parakeet TDT 0.6B v3, q4_k_m with F16
     * convolutions (553 MB). Same WER as q8_0 on FLEURS and the smallest
     * q4-class file; runs on the Vulkan fast engine where available.
     */
    val RECOMMENDED_PARAKEET = ModelDownload(
        displayName = "Parakeet TDT 0.6B v3 (q4_k_m-shrink16)",
        url = "https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf/resolve/" +
            "96402b32bd374742aa1da3c66af30aa64cea3fdb/parakeet-tdt-0.6b-v3-q4_k_m-shrink16.gguf",
        sizeBytes = 552_670_624L,
        sha256 = "2b5ea37e3193c71b3ad2f859b4238ff4501898ba73d63899568faee1daae9982",
    )
}
