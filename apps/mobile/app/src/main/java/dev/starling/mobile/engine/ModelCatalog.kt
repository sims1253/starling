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
     * The recommended on-device model: Parakeet TDT 0.6B v3 with every linear,
     * including the pointwise convolutions, at Q4_0 and the joint/LSTM at Q8_0
     * (364 MB; quants/recipes/parakeet-q4pw-fullimx.recipe). Against the
     * earlier q4_k_m-shrink16 file on a Pixel 10 Pro fast engine: 34 % smaller,
     * 34 % less memory, ~10 % faster, same FLEURS WER.
     */
    val RECOMMENDED_PARAKEET = ModelDownload(
        url = "https://huggingface.co/scholzmx/parakeet-tdt-0.6b-v3-gguf/resolve/" +
            "07008403b0ae5e49632e5d65df264933d494a281/parakeet-tdt-0.6b-v3-q4pw-fullimx.gguf",
        sizeBytes = 364_020_736L,
        sha256 = "6975b243883552a41abac56fd4021aef8ec57489de8fb8854c464fb41a551e26",
        label = "Parakeet TDT 0.6B v3",
        kind = ModelKind.PARAKEET,
    )

    /**
     * Underdog Bark 3.0 Pro (ConwayResearch, CC-BY-4.0): an English
     * fine-tune of Parakeet TDT 0.6B v3 in the same q4pw layout, so the same
     * size and speed. FLEURS en_us test (100 clips) WER 3.68 % vs
     * Parakeet's 4.29 % (Whisper normalization;
     * scripts/convert_parakeet_gguf.py --model-id).
     */
    val BARK_3_PRO = ModelDownload(
        url = "https://huggingface.co/scholzmx/underdog-bark-3.0-pro-gguf/resolve/" +
            "6b96418d56b68646807cdc71ee4c0a4222af2d0a/underdog-bark-3.0-pro-q4pw.gguf",
        sizeBytes = 364_020_736L,
        sha256 = "2370bf360720b322a54bdc4cfe6198713b347e79ebec7853956ade93618d5986",
        label = "Bark 3.0 Pro (English)",
        kind = ModelKind.PARAKEET,
    )

    /**
     * Underdog Bark 3.0 Clean: Bark 3.0 Plus trained to leave out fillers,
     * false starts and repeats while it transcribes (not verbatim). Converted
     * exactly from its entropy-coded model.bark (convert_parakeet_gguf.py --bark),
     * then q4pw.
     */
    val BARK_3_CLEAN = ModelDownload(
        url = "https://huggingface.co/scholzmx/underdog-bark-3.0-clean-gguf/resolve/" +
            "000c6b353d237ff647644ea2c3beb3910a21215b/underdog-bark-3.0-clean-q4pw.gguf",
        sizeBytes = 364_020_736L,
        sha256 = "d1eec8fc4b9aab0fb6a83ddba6a56a0f93b094d350d069f885a843e31eaee0cf",
        label = "Bark 3.0 Clean (English, no fillers)",
        kind = ModelKind.PARAKEET,
    )

    /**
     * Underdog Bark 3.0 Plus: the verbatim model Clean builds on (model.bark,
     * q4pw). Sometimes writes lowercase, unpunctuated text and spelled-out numbers.
     */
    val BARK_3_PLUS = ModelDownload(
        url = "https://huggingface.co/scholzmx/underdog-bark-3.0-plus-gguf/resolve/" +
            "1339fa59186ebacf60a40419fcd53643b2df2658/underdog-bark-3.0-plus-q4pw.gguf",
        sizeBytes = 364_020_736L,
        sha256 = "586e2d0306a82cf87c6d47e1fff7b80484a4143eaf5afb7da9b4a87cbcab3d29",
        label = "Bark 3.0 Plus (English)",
        kind = ModelKind.PARAKEET,
    )

    /**
     * Underdog Bark 3.0: the smallest download of the family upstream; on the
     * phone the same q4pw size as the others. Same output-style caveat as Plus.
     */
    val BARK_3 = ModelDownload(
        url = "https://huggingface.co/scholzmx/underdog-bark-3.0-gguf/resolve/" +
            "01b1d23afaaaf9c132c6d974269e91d01c84e915/underdog-bark-3.0-q4pw.gguf",
        sizeBytes = 364_020_736L,
        sha256 = "5650191831a0b6915003b34a8c82b45ba91d60bf6b36cfa0f08babe43a556081",
        label = "Bark 3.0 (English)",
        kind = ModelKind.PARAKEET,
    )

    /**
     * Qwen3-ASR-0.6B: Q8_0 encoder, Q5_K/Q6_K decoder, Q6_K head
     * (quants/recipes/qwen3-0.6b-k56x8e6.recipe).
     */
    val QWEN3_ASR_06B = ModelDownload(
        url = "https://huggingface.co/scholzmx/qwen3-asr-0.6b-gguf/resolve/" +
            "c6d02fcaac1025e7ed03ad71437b863d8c78c50e/qwen3-asr-0.6b-k56x8e6.gguf",
        sizeBytes = 662_750_912L,
        sha256 = "d7353fb5238ff1bf3df49105c6675d677334535d7681d7ed26c0ef4e6d6616aa",
        label = "Qwen3-ASR 0.6B",
        kind = ModelKind.QWEN3,
    )

    /**
     * ARK-ASR-0.6B: Q4_K encoder, Q4_0/Q8_0 decoder, Q8_0 head
     * (quants/recipes/ark06-kds8.recipe).
     */
    val ARK_ASR_06B = ModelDownload(
        url = "https://huggingface.co/scholzmx/ark-asr-0.6b-gguf/resolve/" +
            "1adeae821a29305e04d0335d0731746d684d75a3/ark-asr-0.6b-kds8.gguf",
        sizeBytes = 805_586_304L,
        sha256 = "2d313f6704a877efc172ce7d06634081c7134a39b50c1cbb75ce4edbfadaf512",
        label = "ARK-ASR 0.6B",
        kind = ModelKind.ARK,
    )

    /**
     * Granite Speech 4.1 2B: Q4_0 throughout, including the conformer convs
     * (quants/recipes/granite-q40-h4.recipe).
     */
    val GRANITE_SPEECH_2B = ModelDownload(
        url = "https://huggingface.co/scholzmx/granite-speech-4.1-2b-gguf/resolve/" +
            "0f8139662734ce0b0caef73994eef99eb8454ea4/granite-speech-4.1-2b-q40-h4.gguf",
        sizeBytes = 1_471_787_424L,
        sha256 = "c6b3fa9cc275f34f55dff5c39ae2c912495e5aeb1190e943d2feda52021097d6",
        label = "Granite Speech 4.1 2B",
        kind = ModelKind.GRANITE,
    )

    /**
     * Qwen3-ASR-1.7B: Q6_K encoder, Q5_K/Q6_K decoder, Q6_K head
     * (quants/recipes/qwen3-1.7b-k56x6e6.recipe).
     */
    val QWEN3_ASR_17B = ModelDownload(
        url = "https://huggingface.co/scholzmx/qwen3-asr-1.7b-gguf/resolve/" +
            "b3c56b1e326637b52247100cc9b2ef52b0945af1/qwen3-asr-1.7b-k56x6e6.gguf",
        sizeBytes = 1_566_998_208L,
        sha256 = "d9f6e53ceecf747f7370baa1795cdf54a6b998648a937d2cc7618bd075e42b23",
        label = "Qwen3-ASR 1.7B",
        kind = ModelKind.QWEN3,
    )

    /**
     * MOSS-Transcribe-preview-2B: Q4_0 linears and tied head
     * (quants/recipes/moss-q4e4-cal.recipe); runs on the Vulkan fast engine
     * where available. English: it mostly translates other languages.
     */
    val MOSS_TRANSCRIBE_2B = ModelDownload(
        url = "https://huggingface.co/scholzmx/moss-transcribe-preview-2b-gguf/resolve/" +
            "44e381cfe2db023adb58503f93cc90a3cc31d1c6/moss-transcribe-preview-2b-q4e4-cal.gguf",
        sizeBytes = 1_397_410_624L,
        sha256 = "380b4e1d34de22511196d62de20da52c2b698e046f062b5eee7db810564d4066",
        label = "MOSS-Transcribe 2B (English)",
        kind = ModelKind.MOSS,
    )

    /**
     * Every catalog model, the recommended one first; their file names are
     * reserved for verified downloads (an import never takes one). Every
     * entry pins a repository revision with its size and SHA-256 (see
     * ModelCatalogTest). ARK-ASR-3B stays import-only: it throttles past
     * real time on a phone CPU.
     */
    val ALL = listOf(
        RECOMMENDED_PARAKEET,
        BARK_3_PRO,
        BARK_3_CLEAN,
        BARK_3_PLUS,
        BARK_3,
        QWEN3_ASR_06B,
        ARK_ASR_06B,
        GRANITE_SPEECH_2B,
        QWEN3_ASR_17B,
        MOSS_TRANSCRIBE_2B,
    )

    /** The catalog entry installed as [fileName], if any. */
    fun forFileName(fileName: String): ModelDownload? = ALL.firstOrNull { it.fileName == fileName }
}
