package dev.starling.mobile.data

/**
 * A recording is kept on disk until the user explicitly deletes it. The raw
 * transcript returned by Starling is stored verbatim in [rawTranscript].
 */
enum class RecordingStatus {
    RECORDING,
    PENDING,
    TRANSCRIBING,
    TRANSCRIBED,
    FAILED,
}

/**
 * How the stored transcript was produced. Provenance only records what
 * happened; it never changes the text itself.
 */
enum class TranscriptionProvenance {
    /** Live `WS /stream` session: partials while recording, final on commit. */
    LIVE_STREAM,

    /** Multipart upload of the saved WAV after the recording finished. */
    BATCH_UPLOAD,
}

data class Recording(
    val id: String,
    val createdAtMillis: Long,
    val wavName: String,
    val status: RecordingStatus,
    val durationSeconds: Double = 0.0,
    val rawTranscript: String? = null,
    val errorMessage: String? = null,
    val attempts: Int = 0,
    val provenance: TranscriptionProvenance? = null,
    /**
     * A take dictated into a private field (password, incognito). It never
     * appears in the history, and the keyboard deletes it as soon as the
     * take settles; any left behind by a crash are deleted on the next start.
     */
    val ephemeral: Boolean = false,
)
